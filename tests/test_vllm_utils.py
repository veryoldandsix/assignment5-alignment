import builtins
import errno
import io
import json
import sys
import urllib.error
import urllib.request
from threading import Event
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from cs336_alignment import vllm_utils


def _missing_route(url):
    return urllib.error.HTTPError(
        url, 404, "Not Found", {}, io.BytesIO(b'{"detail":"Not Found"}')
    )


def test_server_uses_selected_port_for_startup_and_requests(monkeypatch):
    process = Mock()
    select_port = Mock(return_value=45678)
    launch = Mock(return_value=process)
    wait = Mock()
    monkeypatch.setattr(vllm_utils, "_get_available_port", select_port)
    monkeypatch.setattr(vllm_utils, "start_server", launch)
    monkeypatch.setattr(vllm_utils, "wait_for_server", wait)
    monkeypatch.setattr(vllm_utils.atexit, "register", Mock())

    server = vllm_utils.VLLMServer("local-model")
    server.start()

    select_port.assert_called_once_with("127.0.0.1", 0)
    assert server.port == 45678
    assert server.base_url == "http://127.0.0.1:45678"
    assert launch.call_args.kwargs["port"] == 45678
    wait.assert_called_once_with(server.base_url, process, server.startup_timeout)
    generate = Mock(return_value=[])
    monkeypatch.setattr(vllm_utils, "generate_completions", generate)
    server.generate_completions(["question"], {})
    assert generate.call_args.kwargs["vllm_base_url"] == server.base_url


def test_available_port_is_selected_by_operating_system(monkeypatch):
    socket_context = MagicMock()
    listener = socket_context.__enter__.return_value
    listener.getsockname.return_value = ("127.0.0.1", 45678)
    monkeypatch.setattr(vllm_utils.socket, "socket", Mock(return_value=socket_context))

    assert vllm_utils._get_available_port("127.0.0.1", 0) == 45678
    listener.bind.assert_called_once_with(("127.0.0.1", 0))
    socket_context.__exit__.assert_called_once()


def test_occupied_explicit_port_does_not_launch_or_kill_processes(monkeypatch):
    socket_context = MagicMock()
    socket_context.__enter__.return_value.bind.side_effect = OSError(
        errno.EADDRINUSE, "Address already in use"
    )
    monkeypatch.setattr(vllm_utils.socket, "socket", Mock(return_value=socket_context))
    launch = Mock()
    run = Mock()
    monkeypatch.setattr(vllm_utils, "start_server", launch)
    monkeypatch.setattr(vllm_utils.subprocess, "run", run)
    server = vllm_utils.VLLMServer("local-model", port=8000)

    with pytest.raises((RuntimeError, OSError)):
        server.start()

    launch.assert_not_called()
    run.assert_not_called()


def test_launch_uses_current_python_and_enables_weight_transfer(monkeypatch):
    popen = Mock()
    monkeypatch.setattr(vllm_utils.subprocess, "Popen", popen)
    monkeypatch.setenv("VLLM_SERVER_DEV_MODE", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    assert vllm_utils.start_server(
        model_id="local-model",
        host="127.0.0.1",
        port=45678,
        gpu=1,
        seed=7,
        load_format="auto",
        logging_level="ERROR",
    ) is popen.return_value

    command = popen.call_args.args[0]
    assert command[:4] == [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve"]
    assert command[4] == "local-model"
    assert command[command.index("--port") + 1] == "45678"
    assert json.loads(command[command.index("--weight-transfer-config") + 1]) == {"backend": "nccl"}
    env = popen.call_args.kwargs["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["VLLM_SERVER_DEV_MODE"] == "1"
    assert popen.call_args.kwargs["start_new_session"] is True


def test_health_without_weight_transfer_routes_is_not_ready(monkeypatch):
    base_url = "http://127.0.0.1:45678"

    def http_json(method, url, *args, **kwargs):
        assert method == "GET"
        if url == f"{base_url}/health":
            return {}
        assert url == f"{base_url}/get_world_size"
        raise _missing_route(url)

    monkeypatch.setattr(vllm_utils, "_http_json", http_json)
    monkeypatch.setattr(vllm_utils.time, "sleep", Mock(side_effect=AssertionError("Do not retry an incompatible server")))

    with pytest.raises(RuntimeError) as error:
        vllm_utils.wait_for_server(base_url, None, timeout=30)

    assert "/get_world_size" in str(error.value)
    assert "VLLM_SERVER_DEV_MODE" in str(error.value)


def test_startup_failure_cleans_up_its_own_process(monkeypatch):
    process = Mock()
    stop = Mock()
    monkeypatch.setattr(vllm_utils, "_get_available_port", Mock(return_value=45678))
    monkeypatch.setattr(vllm_utils, "start_server", Mock(return_value=process))
    monkeypatch.setattr(vllm_utils, "wait_for_server", Mock(side_effect=TimeoutError("not ready")))
    monkeypatch.setattr(vllm_utils, "stop_server", stop)
    monkeypatch.setattr(vllm_utils.atexit, "register", Mock())
    server = vllm_utils.VLLMServer("local-model")

    with pytest.raises(TimeoutError, match="not ready"):
        server.start()

    stop.assert_called_once_with(process, timeout=server.shutdown_timeout)


def test_inference_world_size_returns_positive_integer(monkeypatch):
    http_json = Mock(return_value={"world_size": 2})
    monkeypatch.setattr(vllm_utils, "_http_json", http_json)

    assert vllm_utils._get_inference_world_size("http://127.0.0.1:45678") == 2
    assert http_json.call_args.args[:2] == ("GET", "http://127.0.0.1:45678/get_world_size")


@pytest.mark.parametrize("payload", [{}, {"world_size": 0}, {"world_size": -1}, {"world_size": "2"}, {"world_size": True}])
def test_inference_world_size_rejects_invalid_server_response(monkeypatch, payload):
    monkeypatch.setattr(vllm_utils, "_http_json", Mock(return_value=payload))

    with pytest.raises(RuntimeError, match="world_size"):
        vllm_utils._get_inference_world_size("http://127.0.0.1:45678")


def test_missing_routes_fail_before_importing_vllm_or_touching_cuda(monkeypatch):
    base_url = "http://127.0.0.1:45678"
    monkeypatch.setattr(vllm_utils, "_http_json", Mock(side_effect=_missing_route(f"{base_url}/get_world_size")))
    set_device = Mock()
    monkeypatch.setattr(vllm_utils.torch.cuda, "set_device", set_device)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "vllm" or name.startswith("vllm."):
            raise AssertionError("Check server capabilities before importing GPU dependencies")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(RuntimeError, match="/get_world_size"):
        vllm_utils.init_weight_sync(base_url, "cuda:0")
    set_device.assert_not_called()


def test_http_requests_use_proxy_free_opener(monkeypatch):
    for handler in vllm_utils._HTTP_OPENER.handlers:
        if isinstance(handler, urllib.request.ProxyHandler):
            assert not handler.proxies
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("http_proxy", "http://proxy.invalid:8080")
    response = MagicMock()
    response.__enter__.return_value.read.return_value = b'{"world_size": 1}'
    direct_open = Mock(return_value=response)
    monkeypatch.setattr(vllm_utils._HTTP_OPENER, "open", direct_open)
    monkeypatch.setattr(vllm_utils.urllib.request, "urlopen", Mock(side_effect=AssertionError("Global opener may use proxies")))

    assert vllm_utils._http_json("GET", "http://127.0.0.1:45678/get_world_size", timeout=10) == {"world_size": 1}
    request = direct_open.call_args.args[0]
    assert request.full_url == "http://127.0.0.1:45678/get_world_size"
    assert request.get_method() == "GET"
    assert direct_open.call_args.kwargs["timeout"] == 10


def test_connecting_to_existing_server_preserves_address(monkeypatch):
    select_port = Mock()
    launch = Mock()
    wait = Mock()
    monkeypatch.setattr(vllm_utils, "_get_available_port", select_port)
    monkeypatch.setattr(vllm_utils, "start_server", launch)
    monkeypatch.setattr(vllm_utils, "wait_for_server", wait)
    server = vllm_utils.VLLMServer("local-model", port=45678, launch_server=False)

    server.start()

    select_port.assert_not_called()
    launch.assert_not_called()
    wait.assert_called_once_with("http://127.0.0.1:45678", None, server.startup_timeout)


def test_exited_server_is_reported_before_checking_http(monkeypatch):
    process = Mock()
    process.poll.return_value = 1
    process.returncode = 1
    http_json = Mock()
    monkeypatch.setattr(vllm_utils, "_http_json", http_json)

    with pytest.raises(RuntimeError, match="exited.*1"):
        vllm_utils.wait_for_server("http://127.0.0.1:45678", process, timeout=30)

    http_json.assert_not_called()


def test_weight_sync_uses_matching_small_buffers_and_refreshes_cache(monkeypatch):
    base_url = "http://127.0.0.1:45678"
    policy = vllm_utils.torch.nn.Linear(3, 2)
    group = object()
    update_started = Event()
    send_finished = Event()
    events = []
    captured = {}

    def http_json(method, url, payload=None, timeout=60):
        assert method == "POST"
        endpoint = url.removeprefix(base_url)
        events.append(endpoint)
        if endpoint == "/update_weights":
            captured["update_info"] = payload["update_info"]
            update_started.set()
            assert send_finished.wait(timeout=5), "Trainer did not send weights"
        return {}

    def send_weights(*, iterator, trainer_args):
        assert update_started.wait(timeout=5), "Server did not start receiving weights"
        events.append("send_weights")
        captured["trainer_args"] = trainer_args
        captured["weights"] = list(iterator)
        send_finished.set()

    for name in ("vllm", "vllm.distributed", "vllm.distributed.weight_transfer"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    nccl_module = ModuleType("vllm.distributed.weight_transfer.nccl_engine")
    nccl_module.NCCLTrainerSendWeightsArgs = lambda **kwargs: SimpleNamespace(**kwargs)
    nccl_module.NCCLWeightTransferEngine = SimpleNamespace(trainer_send_weights=send_weights)
    monkeypatch.setitem(sys.modules, nccl_module.__name__, nccl_module)
    set_device = Mock()
    monkeypatch.setattr(vllm_utils.torch.cuda, "set_device", set_device)
    monkeypatch.setattr(vllm_utils, "_http_json", http_json)

    vllm_utils.sync_policy_weights(policy, base_url, group)

    assert events == ["/pause", "/update_weights", "send_weights", "/reset_prefix_cache", "/resume"]
    update_info = captured["update_info"]
    trainer_args = captured["trainer_args"]
    assert update_info["names"] == ["weight", "bias"]
    assert update_info["dtype_names"] == ["float32", "float32"]
    assert update_info["shapes"] == [[2, 3], [2]]
    assert update_info["packed"] is trainer_args.packed is True
    assert update_info["packed_buffer_size_bytes"] == trainer_args.packed_buffer_size_bytes == 64 * 1024 * 1024
    assert update_info["packed_num_buffers"] == trainer_args.packed_num_buffers == 2
    assert trainer_args.group is group
    assert [name for name, _ in captured["weights"]] == ["weight", "bias"]
    assert captured["weights"][0][1] is policy.weight
    assert captured["weights"][1][1] is policy.bias
    set_device.assert_called_once_with(vllm_utils.torch.device("cpu"))
