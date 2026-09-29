from app import history_text, message_text


def test_message_text_accepts_gradio_content_blocks() -> None:
    content = [{"text": "Primeira resposta"}, {"content": "continuação"}]
    assert message_text(content) == "Primeira resposta\ncontinuação"


def test_message_text_accepts_plain_text() -> None:
    assert message_text("Pergunta simples") == "Pergunta simples"


def test_history_text_removes_telemetry() -> None:
    content = "Resposta útil.\n\n_Tempo: 2.64s | Pico de VRAM: 0.49 GiB_"
    assert history_text(content) == "Resposta útil."
