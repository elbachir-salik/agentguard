from agentguard.similarity import inputs_are_similar


def test_same_arguments_different_ids():
    a = {"id": "call_abc", "name": "search_kb", "arguments": '{"query": "refund"}'}
    b = {"id": "call_xyz", "name": "search_kb", "arguments": '{"query": "refund"}'}
    assert inputs_are_similar(a, b) is True


def test_different_arguments():
    a = {"id": "1", "name": "search_kb", "arguments": '{"query": "refund"}'}
    b = {"id": "2", "name": "search_kb", "arguments": '{"query": "shipping"}'}
    assert inputs_are_similar(a, b) is False


def test_dict_vs_json_string_arguments():
    a = {"arguments": {"query": "refund"}}
    b = {"arguments": '{"query": "refund"}'}
    assert inputs_are_similar(a, b) is True
