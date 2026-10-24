# ----------------------------
# ZMQ Policy Client
# ----------------------------

import zmq
import numpy as np


class ZmqPolicyClient:
    def __init__(self, addr: str, timeout_ms: int = 300):
        self.addr = addr
        self.timeout_ms = int(timeout_ms)
        self.ctx = zmq.Context.instance()
        self.sock = None
        self.seq = 0
        self._make_sock()

    def _make_sock(self):
        if self.sock is not None:
            try:
                self.sock.close(linger=0)
            except Exception:
                pass

        s = self.ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        s.setsockopt(zmq.RCVTIMEO, self.timeout_ms)

        # Allow recovery if we miss a reply
        s.setsockopt(zmq.REQ_RELAXED, 1)
        s.setsockopt(zmq.REQ_CORRELATE, 1)

        s.connect(self.addr)
        self.sock = s

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close(linger=0)

    def _roundtrip(self, parts: list[bytes]) -> list[bytes]:
        try:
            self.sock.send_multipart(parts)
            rep = self.sock.recv_multipart()
        except (zmq.Again, zmq.ZMQError) as e:
            self._make_sock()
            raise RuntimeError(f"ZMQ REQ failed/reset: {e}")

        if len(rep) >= 2 and rep[0] == b"ERR":
            raise RuntimeError(rep[1].decode("utf-8", errors="ignore"))
        return rep

    def predict_action(self, jpg_fix: bytes, jpg_wrist: bytes, state: np.ndarray | None = None) -> np.ndarray:
        """
        Legacy action-only request.

        Request:
          [seq_b(4), jpg_fix, jpg_wrist, state_bytes]   # if state is not None
          [seq_b(4), jpg_fix, jpg_wrist]                # if state is None

        Reply:
          [rep_seq(4), act_bytes]
        """
        seq_b = int(self.seq).to_bytes(4, "big")
        self.seq += 1

        parts = [seq_b, jpg_fix, jpg_wrist]
        if state is not None:
            state = np.asarray(state, dtype=np.float32).reshape(-1)
            parts.append(state.tobytes(order="C"))

        rep = self._roundtrip(parts)

        if len(rep) != 2:
            raise RuntimeError(f"Bad reply parts={len(rep)}")

        rep_seq, act_bytes = rep
        if rep_seq != seq_b:
            raise RuntimeError("Sequence mismatch (REQ/REP out of sync)")

        return np.frombuffer(act_bytes, dtype=np.float32).copy()

    def predict_action_chunk(
        self,
        jpg_fix: bytes,
        jpg_wrist: bytes,
        state: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Chunk request.

        Request:
          [b"ACT_CHUNK", seq_b(4), jpg_fix, jpg_wrist, state_bytes]   # if state is not None
          [b"ACT_CHUNK", seq_b(4), jpg_fix, jpg_wrist]                # if state is None

        Reply:
          [rep_seq(4), act_bytes, chunk_shape_bytes, chunk_bytes]

        Returns:
          act:   (A,)
          chunk: (K, A)
        """
        seq_b = int(self.seq).to_bytes(4, "big")
        self.seq += 1

        parts = [b"ACT_CHUNK", seq_b, jpg_fix, jpg_wrist]
        if state is not None:
            state = np.asarray(state, dtype=np.float32).reshape(-1)
            parts.append(state.tobytes(order="C"))

        rep = self._roundtrip(parts)

        if len(rep) != 4:
            raise RuntimeError(f"Bad ACT_CHUNK reply parts={len(rep)}")

        rep_seq, act_bytes, chunk_shape_bytes, chunk_bytes = rep
        if rep_seq != seq_b:
            raise RuntimeError("Sequence mismatch (REQ/REP out of sync)")

        act = np.frombuffer(act_bytes, dtype=np.float32).copy()

        chunk_shape = np.frombuffer(chunk_shape_bytes, dtype=np.int32)
        if chunk_shape.size != 2:
            raise RuntimeError(f"Bad chunk shape payload: {chunk_shape}")

        k, a = int(chunk_shape[0]), int(chunk_shape[1])
        chunk = np.frombuffer(chunk_bytes, dtype=np.float32).copy().reshape(k, a)

        return act, chunk

    def predict_action_chunk_batch(
        self,
        jpg_fix: bytes,
        jpg_wrist: bytes,
        state: np.ndarray | None = None,
        num_samples: int = 32,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Batch chunk request.

        Request:
        [b"ACT_CHUNK_BATCH", seq_b(4), n_samples_i32(4), jpg_fix, jpg_wrist, state_bytes]  # if state is not None
        [b"ACT_CHUNK_BATCH", seq_b(4), n_samples_i32(4), jpg_fix, jpg_wrist]                # if state is None

        Reply:
        [rep_seq(4), act_bytes, chunk_shape_bytes, chunk_bytes, batch_shape_bytes, batch_bytes]

        Returns:
        act:         (A,)
        chunk:       (K, A)
        chunk_batch: (B, K, A)
        """
        if int(num_samples) < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")

        seq_b = int(self.seq).to_bytes(4, "big")
        self.seq += 1

        n_samples_b = np.asarray([int(num_samples)], dtype=np.int32).tobytes()

        parts = [b"ACT_CHUNK_BATCH", seq_b, n_samples_b, jpg_fix, jpg_wrist]
        if state is not None:
            state = np.asarray(state, dtype=np.float32).reshape(-1)
            parts.append(state.tobytes(order="C"))

        rep = self._roundtrip(parts)

        if len(rep) != 6:
            raise RuntimeError(f"Bad ACT_CHUNK_BATCH reply parts={len(rep)}")

        rep_seq, act_bytes, chunk_shape_bytes, chunk_bytes, batch_shape_bytes, batch_bytes = rep
        if rep_seq != seq_b:
            raise RuntimeError("Sequence mismatch (REQ/REP out of sync)")

        act = np.frombuffer(act_bytes, dtype=np.float32).copy()

        chunk_shape = np.frombuffer(chunk_shape_bytes, dtype=np.int32)
        if chunk_shape.size != 2:
            raise RuntimeError(f"Bad chunk shape payload: {chunk_shape}")
        k, a = int(chunk_shape[0]), int(chunk_shape[1])
        chunk = np.frombuffer(chunk_bytes, dtype=np.float32).copy().reshape(k, a)

        batch_shape = np.frombuffer(batch_shape_bytes, dtype=np.int32)
        if batch_shape.size != 3:
            raise RuntimeError(f"Bad chunk batch shape payload: {batch_shape}")
        b, k2, a2 = int(batch_shape[0]), int(batch_shape[1]), int(batch_shape[2])
        if k2 != k or a2 != a:
            raise RuntimeError(
                f"Chunk/chunk_batch shape mismatch: chunk=({k},{a}) batch=({b},{k2},{a2})"
            )

        chunk_batch = np.frombuffer(batch_bytes, dtype=np.float32).copy().reshape(b, k2, a2)
        return act, chunk, chunk_batch


class ZmqControlClient:
    def __init__(self, addr: str, timeout_ms: int = 100):
        self.addr = addr
        self.timeout_ms = int(timeout_ms)
        self.ctx = zmq.Context.instance()
        self.sock = None
        self._make_sock()

    def _make_sock(self):
        if self.sock is not None:
            try:
                self.sock.close(linger=0)
            except Exception:
                pass

        s = self.ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        s.setsockopt(zmq.RCVTIMEO, self.timeout_ms)

        s.setsockopt(zmq.REQ_RELAXED, 1)
        s.setsockopt(zmq.REQ_CORRELATE, 1)

        s.connect(self.addr)
        self.sock = s

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close(linger=0)

    def poll_command(self) -> str | None:
        try:
            self.sock.send_multipart([b"POLL"])
            rep = self.sock.recv_multipart()
        except (zmq.Again, zmq.ZMQError):
            self._make_sock()
            return None

        if len(rep) == 0:
            return None

        tag = rep[0]

        if tag == b"NONE":
            return None

        if tag == b"ERR":
            msg = rep[1].decode("utf-8", errors="ignore") if len(rep) > 1 else "unknown"
            raise RuntimeError(f"Control server error: {msg}")

        return tag.decode("utf-8", errors="ignore")
    
    # Only used in real robot setting to reset the policy server from footpedal (client) -> policy (server)
    def reset_policy(self) -> bool:
        try:
            self.sock.send_multipart([b"RESET_POLICY"])
            rep = self.sock.recv_multipart()
        except (zmq.Again, zmq.ZMQError):
            self._make_sock()
            return False

        if len(rep) == 0:
            return False

        if rep[0] == b"OK":
            return True

        if rep[0] == b"ERR":
            msg = rep[1].decode("utf-8", errors="ignore") if len(rep) > 1 else "unknown"
            raise RuntimeError(f"Control server error: {msg}")

        return False