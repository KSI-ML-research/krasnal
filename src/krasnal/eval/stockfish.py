import re
import subprocess
import time
from dataclasses import dataclass


@dataclass
class StockfishAnalysis:
    bestmove: str
    score_cp: float | None


class StockfishClient:
    def __init__(
        self,
        depth: int | None = 15,
        nodes: int | None = None,
        binary: str = "stockfish",
    ):
        if depth is not None and nodes is not None:
            raise ValueError("Choose exactly one Stockfish analysis limit: depth or nodes")
        if depth is None and nodes is None:
            raise ValueError("StockfishClient requires either depth or nodes")

        self.binary = binary
        self.depth = depth
        self.nodes = nodes
        self._process: subprocess.Popen[str] | None = None

    def get_eval(self, fen: str) -> float | None:
        analysis = self.analyze(fen)
        return analysis.score_cp

    def get_best_move(self, fen: str) -> str:
        analysis = self.analyze(fen)
        return analysis.bestmove

    def analyze(self, fen: str) -> StockfishAnalysis:
        self._ensure_process()
        assert self._process is not None

        self._send_command(f"position fen {fen}")
        self._send_command(self._build_go_command())
        stdout = self._read_until("bestmove", timeout=10)
        return self._parse_go_output(stdout)

    def _ensure_process(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        self._process = subprocess.Popen(
            [self.binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        self._send_command("uci")
        self._read_until("uciok", timeout=5)
        self._send_command("isready")
        self._read_until("readyok", timeout=5)

    def _send_command(self, command: str) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Stockfish process is not available")
        self._process.stdin.write(f"{command}\n")
        self._process.stdin.flush()

    def _build_go_command(self) -> str:
        if self.nodes is not None:
            return f"go nodes {self.nodes}"
        assert self.depth is not None
        return f"go depth {self.depth}"

    def _read_until(self, marker: str, timeout: float) -> str:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("Stockfish process is not available")

        deadline = time.monotonic() + timeout
        lines: list[str] = []

        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Stockfish did not return '{marker}' within {timeout}s")

            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError("Stockfish closed stdout unexpectedly")
            lines.append(line)
            if marker in line:
                return "".join(lines)

    def _parse_go_output(self, stdout: str) -> StockfishAnalysis:
        mate_score: float | None = None
        cp_score: float | None = None
        bestmove: str | None = None

        for line in stdout.split("\n"):
            if "score mate" in line:
                match = re.search(r"score mate (-?\d+)", line)
                if match:
                    mate_score = float(int(match.group(1)))
            elif "score cp" in line:
                match = re.search(r"score cp (-?\d+)", line)
                if match:
                    cp_score = float(match.group(1))
            elif line.startswith("bestmove "):
                parts = line.split()
                if len(parts) >= 2 and parts[1] != "(none)":
                    bestmove = parts[1]

        if bestmove is None:
            raise RuntimeError(f"Stockfish output did not contain a bestmove. stdout:\n{stdout}")

        if mate_score is not None:
            score_cp = 1000.0 - mate_score * 10 if mate_score > 0 else -1000.0 - mate_score * 10
            return StockfishAnalysis(bestmove=bestmove, score_cp=score_cp)

        return StockfishAnalysis(bestmove=bestmove, score_cp=cp_score)

    def close(self) -> None:
        if self._process is None:
            return

        try:
            if self._process.poll() is None:
                self._send_command("quit")
                self._process.wait(timeout=1)
        except Exception:
            self._process.kill()
        finally:
            self._process = None


def get_stockfish_client(
    depth: int | None = 15,
    nodes: int | None = None,
    binary: str = "stockfish",
) -> StockfishClient:
    return StockfishClient(depth=depth, nodes=nodes, binary=binary)
