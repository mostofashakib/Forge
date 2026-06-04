import pytest
from forge.benchmark.task_suite import Task, TaskSuite, DOMAINS, Difficulty


def test_task_registry_has_email_and_project_mgmt():
    suite = TaskSuite()
    domains = suite.domains()
    assert "email" in domains
    assert "project_mgmt" in domains


def test_each_domain_has_five_levels():
    suite = TaskSuite()
    for domain in suite.domains():
        for level in [1, 2, 3, 4, 5]:
            tasks = suite.tasks_for(domain=domain, depth=level)
            assert len(tasks) >= 1, f"{domain} level {level} has no tasks"


def test_task_has_required_fields():
    suite = TaskSuite()
    task = suite.tasks_for(domain="email", depth=1)[0]
    assert isinstance(task.name, str) and task.name
    assert isinstance(task.objective, str) and task.objective
    assert callable(task.success_fn)
    assert isinstance(task.difficulty, int)
    assert 1 <= task.difficulty <= 5


def test_filter_by_max_depth():
    suite = TaskSuite()
    tasks = suite.tasks_for(domain="email", depth=3)
    assert all(t.difficulty <= 3 for t in tasks)


def test_all_tasks_returns_all_domains_and_depths():
    suite = TaskSuite()
    all_tasks = suite.all_tasks(max_depth=5)
    domains_found = {t.domain for t in all_tasks}
    assert "email" in domains_found
    assert "project_mgmt" in domains_found


def test_success_fn_signature():
    suite = TaskSuite()
    for task in suite.all_tasks():
        result = task.success_fn({"state": "example"})
        assert isinstance(result, bool)
