use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

const RESPONSE_TIMEOUT: Duration = Duration::from_secs(2);

struct UciProcess {
    child: Child,
    stdin: ChildStdin,
    output: Receiver<String>,
}

impl UciProcess {
    fn spawn() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_engine-uci"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("start engine-uci process");
        let stdin = child.stdin.take().expect("engine stdin");
        let stdout = child.stdout.take().expect("engine stdout");
        let (output_tx, output) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(line) => {
                        if output_tx.send(line).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        Self {
            child,
            stdin,
            output,
        }
    }

    fn send(&mut self, command: &str) {
        writeln!(self.stdin, "{command}").expect("write UCI command");
        self.stdin.flush().expect("flush UCI command");
    }

    fn handshake(&mut self) {
        self.send("uci");
        self.expect_line_matching(RESPONSE_TIMEOUT, |line| {
            line == "option name SyzygyPath type string default"
        });
        self.expect_line("uciok", RESPONSE_TIMEOUT);
        self.send("isready");
        self.expect_line("readyok", RESPONSE_TIMEOUT);
    }

    fn expect_line(&self, expected: &str, timeout: Duration) {
        self.expect_line_matching(timeout, |line| line == expected);
    }

    fn expect_line_starting_with(&self, prefix: &str, timeout: Duration) {
        self.expect_line_matching(timeout, |line| line.starts_with(prefix));
    }

    fn expect_line_matching(&self, timeout: Duration, matches: impl Fn(&str) -> bool) {
        let deadline = Instant::now() + timeout;
        let mut observed = Vec::new();
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let line = self
                .output
                .recv_timeout(remaining)
                .unwrap_or_else(|_| panic!("timed out waiting for UCI output; saw {observed:?}"));
            if matches(&line) {
                return;
            }
            observed.push(line);
        }
    }

    fn expect_no_line(&self, timeout: Duration) {
        match self.output.recv_timeout(timeout) {
            Ok(line) => panic!("unexpected UCI output before stop: {line}"),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                panic!("engine-uci closed stdout before the timeout")
            }
        }
    }

    fn wait_for_exit(&mut self) {
        self.send("quit");
        let deadline = Instant::now() + RESPONSE_TIMEOUT;
        while Instant::now() < deadline {
            if self.child.try_wait().expect("poll engine process").is_some() {
                return;
            }
            thread::sleep(Duration::from_millis(10));
        }
        self.child.kill().expect("kill unresponsive engine process");
        self.child.wait().expect("reap engine process");
        panic!("engine-uci did not exit after quit");
    }
}

impl Drop for UciProcess {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            self.wait_for_exit();
        }
    }
}

#[test]
fn infinite_search_waits_for_stop_even_when_max_depth_is_one() {
    let mut uci = UciProcess::spawn();
    uci.handshake();
    uci.send("setoption name Max Depth value 1");
    uci.send("position startpos");
    uci.send("go infinite");

    uci.expect_no_line(Duration::from_millis(150));

    uci.send("stop");
    uci.expect_line_starting_with("bestmove ", RESPONSE_TIMEOUT);
}

#[test]
fn replacing_an_infinite_search_emits_only_the_replacement_bestmove() {
    let mut uci = UciProcess::spawn();
    uci.handshake();
    uci.send("position startpos");
    uci.send("go infinite");
    uci.send("position startpos moves e2e4");
    uci.send("go depth 1");

    uci.expect_line_starting_with("bestmove ", RESPONSE_TIMEOUT);
    uci.expect_no_line(Duration::from_millis(150));
}
