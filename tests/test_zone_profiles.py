from __future__ import annotations

import json
from pathlib import Path

from app.config.loader import load_zones_config
from app.zones.models import Zone, ZoneProfile


def test_load_zones_config_selects_profile(tmp_path: Path) -> None:
    path = tmp_path / "zones.json"
    path.write_text(
        json.dumps(
            {
                "active_profile": "image",
                "profiles": {
                    "image": {
                        "name": "Image profile",
                        "reference_width": 900,
                        "reference_height": 1600,
                        "coordinate_mode": "normalized",
                        "zones": [
                            {
                                "zone_id": "top",
                                "name": "Top",
                                "x1": 0.1,
                                "y1": 0.2,
                                "x2": 0.3,
                                "y2": 0.4,
                                "allowed_classes": ["bottle"],
                            }
                        ],
                    },
                    "video": {
                        "name": "Video profile",
                        "reference_width": 1920,
                        "reference_height": 1080,
                        "coordinate_mode": "pixels",
                        "zones": [
                            {
                                "zone_id": "middle",
                                "name": "Middle",
                                "x1": 10,
                                "y1": 20,
                                "x2": 30,
                                "y2": 40,
                                "allowed_classes": ["bottle"],
                            }
                        ],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_zones_config(path, "video")

    assert config.active_profile == "video"
    assert config.profile.profile_id == "video"
    assert config.profile.zones[0].zone_id == "middle"


def test_zone_profile_converts_normalized_zones_to_pixels() -> None:
    profile = ZoneProfile(
        profile_id="image",
        name="Image",
        reference_width=900,
        reference_height=1600,
        coordinate_mode="normalized",
        zones=[Zone("top", "Top", 0.1, 0.2, 0.3, 0.4, ["bottle"])],
    )

    zone = profile.to_pixel_zones(width=1000, height=500)[0]

    assert zone.x1 == 100
    assert zone.y1 == 100
    assert zone.x2 == 300
    assert zone.y2 == 200
