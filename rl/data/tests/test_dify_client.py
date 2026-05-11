from rl.data.dify_client import _sign_params


def test_sign_params_shape():
    params = _sign_params(
        method="POST",
        url="https://api-bj.clink.cn/agent/v1/create-conversation",
        access_key_id="test-ak",
        access_key_secret="test-sk",
    )
    assert set(params.keys()) == {"AccessKeyId", "Expires", "Timestamp", "Signature"}
    assert params["AccessKeyId"] == "test-ak"
    assert params["Signature"]

