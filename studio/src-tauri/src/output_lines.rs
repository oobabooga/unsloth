use std::io::{BufRead, Read};

// Child protocols put control messages on short physical lines. Bounding every
// stream at the source keeps diagnostics and IPC payloads finite, while draining
// an oversized line preserves the next control message as an independent line.
pub(crate) const MAX_LINE_BYTES: usize = 16 * 1024;
pub(crate) const TRUNCATION_MARKER: &str = "… [line truncated]";

pub(crate) struct LossyLine {
    pub(crate) text: String,
    pub(crate) truncated: bool,
}

pub(crate) fn trim_line_endings(bytes: &[u8]) -> &[u8] {
    let mut end = bytes.len();
    while end > 0 && matches!(bytes[end - 1], b'\n' | b'\r') {
        end -= 1;
    }
    &bytes[..end]
}

pub(crate) fn lossy_line(bytes: &[u8], truncated: bool) -> LossyLine {
    let mut text = String::from_utf8_lossy(trim_line_endings(bytes)).into_owned();
    if truncated {
        text.push_str(TRUNCATION_MARKER);
    }
    LossyLine { text, truncated }
}

pub(crate) fn read_lossy_lines<R: Read>(
    stream: R,
    mut on_line: impl FnMut(LossyLine),
) -> std::io::Result<()> {
    let mut reader = std::io::BufReader::new(stream);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match read_bounded_line(&mut reader, &mut buf)? {
            (0, _) => return Ok(()),
            (_, truncated) => on_line(lossy_line(&buf, truncated)),
        }
    }
}

pub(crate) fn read_bounded_line<R: BufRead>(
    reader: &mut R,
    buf: &mut Vec<u8>,
) -> std::io::Result<(usize, bool)> {
    let mut limited = Read::take(&mut *reader, (MAX_LINE_BYTES + 1) as u64);
    let bytes_read = limited.read_until(b'\n', buf)?;
    let truncated = buf.len() > MAX_LINE_BYTES && !buf.ends_with(b"\n");
    if truncated {
        buf.truncate(MAX_LINE_BYTES);
        loop {
            let available = match reader.fill_buf() {
                Ok(available) => available,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            };
            if available.is_empty() {
                break;
            }
            if let Some(index) = available.iter().position(|byte| *byte == b'\n') {
                reader.consume(index + 1);
                break;
            }
            let available_len = available.len();
            reader.consume(available_len);
        }
    }
    Ok((bytes_read, truncated))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    struct InterruptSecondFill<R> {
        inner: R,
        fill_calls: usize,
    }

    impl<R: Read> Read for InterruptSecondFill<R> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.inner.read(buf)
        }
    }

    impl<R: BufRead> BufRead for InterruptSecondFill<R> {
        fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
            self.fill_calls += 1;
            if self.fill_calls == 2 {
                return Err(std::io::Error::from(std::io::ErrorKind::Interrupted));
            }
            self.inner.fill_buf()
        }

        fn consume(&mut self, amount: usize) {
            self.inner.consume(amount);
        }
    }

    #[test]
    fn oversized_lines_are_bounded_and_drain_retries_interrupts() {
        let mut input = vec![b'a'; MAX_LINE_BYTES + 100];
        input.extend_from_slice(b"\nnext\n");
        let mut reader = InterruptSecondFill {
            inner: Cursor::new(input),
            fill_calls: 0,
        };
        let mut line = Vec::new();

        let (_, truncated) = read_bounded_line(&mut reader, &mut line).unwrap();
        assert!(truncated);
        assert_eq!(line.len(), MAX_LINE_BYTES);

        line.clear();
        let (_, truncated) = read_bounded_line(&mut reader, &mut line).unwrap();
        assert!(!truncated);
        assert_eq!(line, b"next\n");
    }
}
