from dataclasses import dataclass


@dataclass
class Datapoint:
    claim: str
    claim_id: int
    claim_date: str = None
    speaker: str = None
    original_claim_url: str = None
    reporting_source: str = None
    location_ISO_code: str = None
    label: str = None
    metadata: dict = None

    @classmethod
    def from_dict(cls, json_data: dict, claim_id: int = None):
        json_data = json_data.copy()
        return cls(
            claim=json_data.pop("claim"),
            claim_id=json_data.pop("claim_id", claim_id),
            claim_date=json_data.pop("claim_date", None),
            speaker=json_data.pop("speaker", None),
            original_claim_url=json_data.pop("original_claim_url", None),
            reporting_source=json_data.pop("reporting_source", None),
            location_ISO_code=json_data.pop("location_ISO_code", None),
            label=json_data.pop("label", None),
            metadata=json_data,
        )

    def to_dict(self):
        return {
            "claim": self.claim,
            "claim_id": self.claim_id,
            "claim_date": self.claim_date,
            "speaker": self.speaker,
            "original_claim_url": self.original_claim_url,
            "reporting_source": self.reporting_source,
            "location_ISO_code": self.location_ISO_code,
            "label": self.label,
            **self.metadata,
        }
