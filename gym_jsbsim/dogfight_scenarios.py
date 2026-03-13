from dataclasses import dataclass

from gym_jsbsim import properties as prp


@dataclass(frozen=True)
class PlaneStart:
    north_offset_m: float = 0.0
    east_offset_m: float = 0.0
    altitude_offset_ft: float = 0.0
    heading_deg: float = 0.0


@dataclass(frozen=True)
class DogfightScenario:
    name: str
    description: str
    plane_a: PlaneStart
    plane_b: PlaneStart


def _plane(north_m: float, east_m: float, heading_deg: float, altitude_ft: float = 0.0) -> PlaneStart:
    return PlaneStart(
        north_offset_m=float(north_m),
        east_offset_m=float(east_m),
        altitude_offset_ft=float(altitude_ft),
        heading_deg=float(heading_deg),
    )


SCENARIOS = (
    DogfightScenario(
        name="head_on_500m",
        description="Head-on merge, 500 m separation.",
        plane_a=_plane(-250.0, 0.0, 0.0),
        plane_b=_plane(250.0, 0.0, 180.0),
    ),
    DogfightScenario(
        name="head_on_1000m",
        description="Head-on merge, 1000 m separation.",
        plane_a=_plane(-500.0, 0.0, 0.0),
        plane_b=_plane(500.0, 0.0, 180.0),
    ),
    DogfightScenario(
        name="head_on_offset_right_1200m",
        description="Head-on merge with 300 m lateral offset to the right.",
        plane_a=_plane(-600.0, -150.0, 0.0),
        plane_b=_plane(600.0, 150.0, 180.0),
    ),
    DogfightScenario(
        name="plane_a_behind_600m",
        description="Plane A starts directly behind plane B by 600 m.",
        plane_a=_plane(-300.0, 0.0, 0.0),
        plane_b=_plane(300.0, 0.0, 0.0),
    ),
    DogfightScenario(
        name="plane_a_behind_1200m",
        description="Plane A starts directly behind plane B by 1200 m.",
        plane_a=_plane(-600.0, 0.0, 0.0),
        plane_b=_plane(600.0, 0.0, 0.0),
    ),
    DogfightScenario(
        name="plane_b_behind_600m",
        description="Plane B starts directly behind plane A by 600 m.",
        plane_a=_plane(300.0, 0.0, 0.0),
        plane_b=_plane(-300.0, 0.0, 0.0),
    ),
    DogfightScenario(
        name="plane_b_behind_1200m",
        description="Plane B starts directly behind plane A by 1200 m.",
        plane_a=_plane(600.0, 0.0, 0.0),
        plane_b=_plane(-600.0, 0.0, 0.0),
    ),
    DogfightScenario(
        name="crossing_left_close",
        description="90-degree crossing, plane B crosses from plane A's left at close range.",
        plane_a=_plane(-350.0, 0.0, 0.0),
        plane_b=_plane(0.0, -350.0, 90.0),
    ),
    DogfightScenario(
        name="crossing_right_close",
        description="90-degree crossing, plane B crosses from plane A's right at close range.",
        plane_a=_plane(-350.0, 0.0, 0.0),
        plane_b=_plane(0.0, 350.0, 270.0),
    ),
    DogfightScenario(
        name="crossing_left_far",
        description="90-degree crossing, plane B crosses from plane A's left at longer range.",
        plane_a=_plane(-650.0, 0.0, 0.0),
        plane_b=_plane(0.0, -650.0, 90.0),
    ),
    DogfightScenario(
        name="crossing_right_far",
        description="90-degree crossing, plane B crosses from plane A's right at longer range.",
        plane_a=_plane(-650.0, 0.0, 0.0),
        plane_b=_plane(0.0, 650.0, 270.0),
    ),
    DogfightScenario(
        name="line_abreast_left_800m",
        description="Same heading, plane B is 800 m off plane A's left wing.",
        plane_a=_plane(0.0, 400.0, 0.0),
        plane_b=_plane(0.0, -400.0, 0.0),
    ),
    DogfightScenario(
        name="line_abreast_right_800m",
        description="Same heading, plane B is 800 m off plane A's right wing.",
        plane_a=_plane(0.0, -400.0, 0.0),
        plane_b=_plane(0.0, 400.0, 0.0),
    ),
    DogfightScenario(
        name="vertical_stack_a_high_500ft",
        description="Co-located start with plane A 500 ft above plane B.",
        plane_a=_plane(0.0, 0.0, 0.0, 250.0),
        plane_b=_plane(0.0, 0.0, 180.0, -250.0),
    ),
    DogfightScenario(
        name="vertical_stack_b_high_500ft",
        description="Co-located start with plane B 500 ft above plane A.",
        plane_a=_plane(0.0, 0.0, 180.0, -250.0),
        plane_b=_plane(0.0, 0.0, 0.0, 250.0),
    ),
    DogfightScenario(
        name="high_aspect_tail_chase_left",
        description="Plane A starts behind while offset left for an oblique tail chase.",
        plane_a=_plane(-500.0, -180.0, 10.0),
        plane_b=_plane(500.0, 0.0, 15.0),
    ),
    DogfightScenario(
        name="high_aspect_tail_chase_right",
        description="Plane B starts behind while offset right for an oblique tail chase.",
        plane_a=_plane(500.0, 0.0, 345.0),
        plane_b=_plane(-500.0, 180.0, 350.0),
    ),
    DogfightScenario(
        name="descending_merge_1000m",
        description="Head-on merge with 1000 ft vertical split.",
        plane_a=_plane(-500.0, 0.0, 0.0, 500.0),
        plane_b=_plane(500.0, 0.0, 180.0, -500.0),
    ),
)


SCENARIO_BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}


def list_scenarios() -> list[str]:
    return [scenario.name for scenario in SCENARIOS]


def get_scenario(name: str) -> DogfightScenario:
    try:
        return SCENARIO_BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"unknown dogfight scenario: {name}") from exc


def scenario_initial_conditions(scenario: DogfightScenario) -> dict[str, dict]:
    return {
        "plane_a": {prp.initial_heading_deg: scenario.plane_a.heading_deg},
        "plane_b": {prp.initial_heading_deg: scenario.plane_b.heading_deg},
    }
