from forge.extraction.entity_extractor import EntityExtractor, _EntityExtractionResult
from forge.extraction.llm_client import MockLLMClient
from forge.extraction.schemas import EntityDef, FieldDef


def _mock_entities() -> _EntityExtractionResult:
    return _EntityExtractionResult(entities=[
        EntityDef(name="ticket", fields=[
            FieldDef(name="id", type="string"),
            FieldDef(name="status", type="enum", values=["open", "closed"]),
        ])
    ])


def test_extractor_returns_entity_list():
    client = MockLLMClient({"_EntityExtractionResult": _mock_entities()})
    extractor = EntityExtractor(client)
    entities = extractor.extract("A ticketing system")
    assert len(entities) == 1
    assert entities[0].name == "ticket"


def test_extractor_passes_prompt_to_llm():
    calls = []

    class SpyClient:
        def extract(self, system, user, schema):
            calls.append(user)
            return _mock_entities()

    EntityExtractor(SpyClient()).extract("my special prompt")
    assert "my special prompt" in calls[0]
