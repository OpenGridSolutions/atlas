// SPDX-License-Identifier: AGPL-3.0-only

//! [`DirectRows`]: table rows read with O_DIRECT, batched through io_uring.
//!
//! Every row is fetched as the 4 KiB block(s) that contain it (O_DIRECT needs
//! aligned offsets, sizes and buffers); a row never spans more than two
//! blocks because rows are at most one block long. Reads are submitted
//! [`QUEUE_DEPTH`] at a time. Where the kernel refuses io_uring (the default
//! seccomp profile of container runtimes does), the same reads run as
//! `pread`s on [`PREAD_WORKERS`] threads — see [`DirectRows::engine`].

use anyhow::{Context, Result, bail};
use std::alloc::{Layout, alloc, dealloc};
use std::collections::HashMap;
use std::fs::File;
use std::os::fd::{AsRawFd, RawFd};
use std::path::PathBuf;

use super::{RowPlane, RowRead, RowSource, checked_len};

const BLOCK: usize = 4096;
/// A row of up to one block can straddle one block boundary.
const WINDOW: usize = 2 * BLOCK;
const QUEUE_DEPTH: usize = 64;
/// Threads of the `pread` engine. An uncached 4 KiB read takes about 0.2 ms
/// on the GB10's NVMe; a decoded token needs 32 of them.
const PREAD_WORKERS: usize = 16;

/// [`QUEUE_DEPTH`] block-aligned read windows in one allocation.
struct Windows {
    ptr: *mut u8,
    layout: Layout,
}

// SAFETY: `Windows` uniquely owns its allocation (created in `new`, freed in
// `Drop` with the same layout) and hands out access only through `&mut self`.
unsafe impl Send for Windows {}

impl Windows {
    fn new() -> Result<Self> {
        let layout = Layout::from_size_align(QUEUE_DEPTH * WINDOW, BLOCK)?;
        // SAFETY: the layout has non-zero size.
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            bail!("out of memory allocating the row read windows");
        }
        Ok(Self { ptr, layout })
    }

    fn ptr(&mut self, slot: usize) -> *mut u8 {
        debug_assert!(slot < QUEUE_DEPTH);
        // SAFETY: slot < QUEUE_DEPTH keeps the offset inside the allocation.
        unsafe { self.ptr.add(slot * WINDOW) }
    }

    fn bytes(&self, slot: usize, start: usize, len: usize) -> &[u8] {
        debug_assert!(slot < QUEUE_DEPTH && start + len <= WINDOW);
        // SAFETY: the range lies inside window `slot`; the kernel has finished
        // writing it before any caller reads (reads are reaped first).
        unsafe { std::slice::from_raw_parts(self.ptr.add(slot * WINDOW + start), len) }
    }
}

impl Drop for Windows {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `alloc` with `layout`.
        unsafe { dealloc(self.ptr, self.layout) }
    }
}

/// How the reads reach the disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowEngine {
    /// Batched through io_uring.
    IoUring,
    /// `pread`s on a pool of threads (io_uring unavailable in this environment).
    Pread,
}

/// One block-aligned read covering a row.
struct Window {
    fd: RawFd,
    offset: u64,
    len: usize,
    /// Where the row starts inside the window.
    skip: usize,
    row_bytes: usize,
}

/// One `pread` for a worker: fill `len` bytes at `dst` from `fd` at `offset`.
struct PreadJob {
    fd: RawFd,
    dst: *mut u8,
    len: usize,
    offset: u64,
    slot: usize,
}

// SAFETY: `dst` points into a window of `Windows` that no other job of the
// batch uses; `read_batch` waits for every job before it touches the windows
// again or returns, so the pointer never outlives or aliases its buffer.
unsafe impl Send for PreadJob {}

/// Bytes read (or the error) for the job of `slot`.
type PreadDone = (usize, std::io::Result<usize>);

/// Worker threads that run `pread`s concurrently.
struct PreadPool {
    jobs: Option<std::sync::mpsc::Sender<PreadJob>>,
    done: std::sync::mpsc::Receiver<PreadDone>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl PreadPool {
    fn new() -> Result<Self> {
        let (jobs, job_rx) = std::sync::mpsc::channel::<PreadJob>();
        let (done_tx, done) = std::sync::mpsc::channel::<PreadDone>();
        let job_rx = std::sync::Arc::new(std::sync::Mutex::new(job_rx));
        let mut workers = Vec::with_capacity(PREAD_WORKERS);
        for i in 0..PREAD_WORKERS {
            let (job_rx, done_tx) = (job_rx.clone(), done_tx.clone());
            let worker = std::thread::Builder::new()
                .name(format!("row-pread-{i}"))
                .spawn(move || {
                    loop {
                        // Hold the lock only while taking a job.
                        let job = match job_rx.lock() {
                            Ok(rx) => rx.recv(),
                            Err(_) => return,
                        };
                        let Ok(job) = job else { return }; // pool dropped
                        if done_tx.send((job.slot, pread_full(&job))).is_err() {
                            return;
                        }
                    }
                })
                .context("spawning a table read thread")?;
            workers.push(worker);
        }
        Ok(Self {
            jobs: Some(jobs),
            done,
            workers,
        })
    }
}

impl Drop for PreadPool {
    fn drop(&mut self) {
        self.jobs = None; // closes the channel; workers leave their loop
        for w in self.workers.drain(..) {
            w.join().ok();
        }
    }
}

/// Reads until `len` bytes are there or the file ends.
fn pread_full(job: &PreadJob) -> std::io::Result<usize> {
    let mut filled = 0usize;
    while filled < job.len {
        // SAFETY: the destination is the unfilled tail of the job's window,
        // `job.len - filled` bytes long (see `PreadJob`).
        let n = unsafe {
            libc::pread(
                job.fd,
                job.dst.add(filled).cast(),
                job.len - filled,
                (job.offset + filled as u64) as libc::off_t,
            )
        };
        if n < 0 {
            return Err(std::io::Error::last_os_error());
        }
        if n == 0 {
            break; // end of file inside the last window
        }
        filled += n as usize;
    }
    Ok(filled)
}

pub struct DirectRows {
    planes: Vec<RowPlane>,
    /// Index into `files` per plane.
    plane_file: Vec<usize>,
    files: Vec<File>,
    windows: Windows,
    #[cfg(target_os = "linux")]
    ring: Option<io_uring::IoUring>,
    /// Present exactly when `ring` is absent.
    pool: Option<PreadPool>,
}

impl DirectRows {
    /// Opens the files of `planes` with O_DIRECT and checks that every plane
    /// lies inside its file.
    pub fn open(planes: Vec<RowPlane>) -> Result<Self> {
        let mut files = Vec::new();
        let mut by_path: HashMap<PathBuf, usize> = HashMap::new();
        let mut plane_file = Vec::with_capacity(planes.len());
        for (i, p) in planes.iter().enumerate() {
            if p.row_bytes == 0 || p.row_bytes > BLOCK {
                bail!("plane {i}: row size {} outside 1..={BLOCK}", p.row_bytes);
            }
            let idx = match by_path.get(&p.path) {
                Some(&idx) => idx,
                None => {
                    let f = open_direct(&p.path)
                        .with_context(|| format!("opening table file {}", p.path.display()))?;
                    files.push(f);
                    by_path.insert(p.path.clone(), files.len() - 1);
                    files.len() - 1
                }
            };
            let size = files[idx].metadata()?.len();
            let end = p.data_offset + p.rows * p.row_bytes as u64;
            if end > size {
                bail!(
                    "plane {i}: {} rows of {} bytes at offset {} end past {} ({size} bytes)",
                    p.rows,
                    p.row_bytes,
                    p.data_offset,
                    p.path.display()
                );
            }
            plane_file.push(idx);
        }
        #[cfg(target_os = "linux")]
        let ring = match io_uring::IoUring::new(QUEUE_DEPTH as u32) {
            Ok(ring) => Some(ring),
            Err(e) => {
                tracing::warn!(
                    "io_uring is not available ({e}); table rows are read with pread on \
                     {PREAD_WORKERS} threads"
                );
                None
            }
        };
        #[cfg(target_os = "linux")]
        let pool = if ring.is_none() {
            Some(PreadPool::new()?)
        } else {
            None
        };
        #[cfg(not(target_os = "linux"))]
        let pool = Some(PreadPool::new()?);
        Ok(Self {
            planes,
            plane_file,
            files,
            windows: Windows::new()?,
            #[cfg(target_os = "linux")]
            ring,
            pool,
        })
    }

    /// The engine selected at open time.
    pub fn engine(&self) -> RowEngine {
        #[cfg(target_os = "linux")]
        if self.ring.is_some() {
            return RowEngine::IoUring;
        }
        RowEngine::Pread
    }

    /// Forces the `pread` engine (tests compare both engines).
    pub fn use_pread(&mut self) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            self.ring = None;
        }
        if self.pool.is_none() {
            self.pool = Some(PreadPool::new()?);
        }
        Ok(())
    }

    fn window(&self, r: &RowRead) -> Window {
        let p = &self.planes[r.plane as usize];
        let abs = p.data_offset + r.row * p.row_bytes as u64;
        let offset = abs - abs % BLOCK as u64;
        let end = (abs + p.row_bytes as u64).div_ceil(BLOCK as u64) * BLOCK as u64;
        Window {
            fd: self.files[self.plane_file[r.plane as usize]].as_raw_fd(),
            offset,
            len: (end - offset) as usize,
            skip: (abs - offset) as usize,
            row_bytes: p.row_bytes,
        }
    }

    /// Fills the windows of one batch (at most [`QUEUE_DEPTH`]) and returns
    /// the bytes each read delivered.
    fn read_batch(&mut self, batch: &[Window]) -> Result<Vec<usize>> {
        let mut got = vec![0usize; batch.len()];
        #[cfg(target_os = "linux")]
        if let Some(ring) = self.ring.as_mut() {
            use io_uring::{opcode, types};
            for (slot, w) in batch.iter().enumerate() {
                let sqe = opcode::Read::new(types::Fd(w.fd), self.windows.ptr(slot), w.len as u32)
                    .offset(w.offset)
                    .build()
                    .user_data(slot as u64);
                // SAFETY: the buffer is window `slot` of `self.windows`, which
                // outlives the submission: all completions are reaped below
                // before this function returns.
                if unsafe { ring.submission().push(&sqe) }.is_err() {
                    bail!(
                        "io_uring submission queue full with {} of {} reads",
                        slot,
                        batch.len()
                    );
                }
            }
            let mut done = 0;
            while done < batch.len() {
                ring.submit_and_wait(batch.len() - done)
                    .context("io_uring submit")?;
                for cqe in ring.completion() {
                    let slot = cqe.user_data() as usize;
                    if cqe.result() < 0 {
                        bail!(
                            "table row read failed: {}",
                            std::io::Error::from_raw_os_error(-cqe.result())
                        );
                    }
                    got[slot] = cqe.result() as usize;
                    done += 1;
                }
            }
            return Ok(got);
        }
        let Some(pool) = self.pool.as_ref() else {
            bail!("table reader has neither io_uring nor a pread pool");
        };
        let Some(jobs) = pool.jobs.as_ref() else {
            bail!("table read pool is shut down");
        };
        for (slot, w) in batch.iter().enumerate() {
            let job = PreadJob {
                fd: w.fd,
                dst: self.windows.ptr(slot),
                len: w.len,
                offset: w.offset,
                slot,
            };
            jobs.send(job)
                .ok()
                .context("table read threads have exited")?;
        }
        // Every job is collected before returning, also after an error: the
        // workers write into the windows until then.
        let mut first_err = None;
        for _ in 0..batch.len() {
            let (slot, res) = pool.done.recv().context("table read threads have exited")?;
            match res {
                Ok(n) => got[slot] = n,
                Err(e) => first_err = first_err.or(Some(e)),
            }
        }
        if let Some(e) = first_err {
            return Err(e).context("table row pread");
        }
        Ok(got)
    }
}

impl RowSource for DirectRows {
    fn row_bytes(&self, plane: u32) -> Result<usize> {
        match self.planes.get(plane as usize) {
            Some(p) => Ok(p.row_bytes),
            None => bail!("no plane {plane}"),
        }
    }

    fn read_rows(&mut self, reads: &[RowRead], out: &mut [u8]) -> Result<()> {
        let geometry: Vec<(usize, u64)> =
            self.planes.iter().map(|p| (p.row_bytes, p.rows)).collect();
        if checked_len(&geometry, reads)? != out.len() {
            bail!("row output buffer has the wrong length");
        }
        let mut at = 0usize;
        for chunk in reads.chunks(QUEUE_DEPTH) {
            let batch: Vec<Window> = chunk.iter().map(|r| self.window(r)).collect();
            let got = self.read_batch(&batch)?;
            for (slot, w) in batch.iter().enumerate() {
                // The last window of a file may be cut short by its end; the
                // row itself must be complete.
                if got[slot] < w.skip + w.row_bytes {
                    bail!(
                        "short table read: {} bytes at offset {}, row needs {}",
                        got[slot],
                        w.offset,
                        w.skip + w.row_bytes
                    );
                }
                out[at..at + w.row_bytes].copy_from_slice(self.windows.bytes(
                    slot,
                    w.skip,
                    w.row_bytes,
                ));
                at += w.row_bytes;
            }
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn open_direct(path: &std::path::Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT | libc::O_CLOEXEC)
        .open(path)
}

#[cfg(not(target_os = "linux"))]
fn open_direct(_path: &std::path::Path) -> std::io::Result<File> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "O_DIRECT table reads are only supported on Linux",
    ))
}
