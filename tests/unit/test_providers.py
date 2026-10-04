from private_agent.agent.providers import create_local_chat_model


def test_local_model_receives_bounded_context_and_output_options():
    captured = {}

    def model_factory(**options):
        captured.update(options)
        return object()

    create_local_chat_model(
        "local-model",
        temperature=0.1,
        base_url="http://localhost:11434",
        acceleration_mode="auto",
        thinking=None,
        chat_model_factory=model_factory,
        client_kwargs=lambda: {},
        acceleration_options=lambda _mode: {},
        track_clients=lambda _model: None,
        max_output_tokens=512,
        context_window=4096,
    )

    assert captured["num_ctx"] == 4096
    assert captured["num_predict"] == 512
