from visionalert.app import DiscardingQueue


def test_queue_should_discard_oldest_items():
    q = DiscardingQueue(3)
    for i in range(5):
        q.put(i)
    assert q.get() == 2


def test_queue_should_invoke_overflow_action(mocker):
    action = mocker.Mock()
    q = DiscardingQueue(3, overflow_action=action)
    for i in range(5):
        q.put(i)
    assert action.call_count == 2


def test_queue_should_init_semaphore_to_zero():
    q = DiscardingQueue(3)
    assert q._semaphore._value == 0
