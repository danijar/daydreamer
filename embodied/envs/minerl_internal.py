from minerl.herobraine.env_specs import obtain_specs
from minerl.herobraine.env_specs import simple_embodiment
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero import mc


SIZE = (64, 64)


class Axe(obtain_specs.Obtain):

  def __init__(self):
    super().__init__(
        target_item='wooden_axe',
        dense=True,
        reward_schedule=[
            dict(type="log", amount=1, reward=1),
            dict(type="crafting_table", amount=1, reward=1),
            dict(type="wooden_axe", amount=1, reward=10),
        ],
        max_episode_steps=int(1e6),
        resolution=SIZE,
    )
    self.name = 'MinecraftAxe-v1'

  def create_agent_handlers(self):
    return []


class Table(obtain_specs.Obtain):

  def __init__(self):
    super().__init__(
        target_item='crafting_table',
        dense=True,
        reward_schedule=[
            dict(type="log", amount=1, reward=1),
            dict(type="crafting_table", amount=1, reward=10),
        ],
        max_episode_steps=int(1e6),
        resolution=SIZE,
    )
    self.name = 'MinecraftTable-v1'

  def create_agent_handlers(self):
    return []


class Wood(obtain_specs.Obtain):

  def __init__(self):
    super().__init__(
        target_item='log',
        dense=True,
        reward_schedule=[
            dict(type="log", amount=1, reward=10),
        ],
        max_episode_steps=int(1e6),
        resolution=SIZE,
    )
    self.name = 'MinecraftWood-v1'

  def create_agent_handlers(self):
    return []


class Diamond(obtain_specs.Obtain):

  def __init__(self):
    super().__init__(
        target_item='diamond',
        dense=False,
        reward_schedule=[
            dict(type="log", amount=1, reward=1),
            dict(type="planks", amount=1, reward=2),
            dict(type="stick", amount=1, reward=4),
            dict(type="crafting_table", amount=1, reward=4),
            dict(type="wooden_pickaxe", amount=1, reward=8),
            dict(type="cobblestone", amount=1, reward=16),
            dict(type="furnace", amount=1, reward=32),
            dict(type="stone_pickaxe", amount=1, reward=32),
            dict(type="iron_ore", amount=1, reward=64),
            dict(type="iron_ingot", amount=1, reward=128),
            dict(type="iron_pickaxe", amount=1, reward=256),
            dict(type="diamond", amount=1, reward=1024)
        ],
        # The time limit used to be 18000 steps but MineRL did not enforce it
        # exactly. We disable the MineRL time limit to apply our own exact
        # time limit on the outside.
        max_episode_steps=int(1e6),
        resolution=SIZE,
    )
    self.name = 'MinecraftDiamond-v1'

  def create_agent_handlers(self):
    # There used to be a handler that terminates the episode after breaking a
    # diamond block. However, it often did not leave enough time to collect
    # the diamond item and receive a reward, so we just continue the episode.
    return []


class Discover(simple_embodiment.SimpleEmbodimentEnvSpec):

  def __init__(self, fps=20):
    self.fps = fps
    super().__init__(
        name='MinecraftDiscover-v1',
        resolution=SIZE,
        max_episode_steps=int(1e8),
    )

  def create_rewardables(self):
    return []

  def create_agent_start(self):
    return []

  def create_agent_handlers(self):
    return []

  def create_server_world_generators(self):
    return [handlers.DefaultWorldGenerator(force_reset=True)]

  def create_server_quit_producers(self):
    return [handlers.ServerQuitWhenAnyAgentFinishes()]

  def create_server_decorators(self):
    return []

  def create_server_initial_conditions(self):
    return [
        handlers.TimeInitialCondition(
            allow_passage_of_time=True,
            start_time=0,
        ),
        handlers.SpawningInitialCondition(
            allow_spawning=True,
        )
    ]

  def determine_success_from_rewards(self, rewards: list):
    return True

  def is_from_folder(self, folder: str):
    return folder == 'none'

  def get_docstring(self):
    return ''

  def create_mission_handlers(self):
    return []

  def create_observables(self):
    return [
        handlers.POVObservation((64, 64)),
        handlers.FlatInventoryObservation(mc.ALL_ITEMS),
        handlers.EquippedItemObservation(
            mc.ALL_ITEMS, _default='air', _other='other'),
    ]

  def create_actionables(self):
    kw = dict(_other='none', _default='none')
    return super().create_actionables() + [
        handlers.PlaceBlock(['none'] + mc.ALL_ITEMS, **kw),
        handlers.EquipAction(['none'] + mc.ALL_ITEMS, **kw),
        handlers.CraftAction(['none'] + mc.ALL_ITEMS, **kw),
        handlers.CraftNearbyAction(['none'] + mc.ALL_ITEMS, **kw),
        handlers.SmeltItemNearby(['none'] + mc.ALL_ITEMS, **kw),
    ]


NOOP_ACTION = dict(
    camera=(0, 0), forward=0, back=0, left=0, right=0, attack=0, sprint=0,
    jump=0, sneak=0, craft='none', nearbyCraft='none', nearbySmelt='none',
    place='none', equip='none',
)

BASIC_ACTIONS = {
    'noop': dict(),
    'attack': dict(attack=1),
    'turn_up': dict(camera=(-15, 0)),
    'turn_down': dict(camera=(15, 0)),
    'turn_left': dict(camera=(0, -15)),
    'turn_right': dict(camera=(0, 15)),
    'forward': dict(forward=1),
    'back': dict(back=1),
    'left': dict(left=1),
    'right': dict(right=1),
    'jump': dict(jump=1, forward=1),
    'place_dirt': dict(place='dirt'),
}

WOOD_ACTIONS = {
    **BASIC_ACTIONS,
}

TABLE_ACTIONS = {
    **BASIC_ACTIONS,
    'craft_planks': dict(craft='planks'),
    'craft_stick': dict(craft='stick'),
    'craft_crafting_table': dict(craft='crafting_table'),
    'place_crafting_table': dict(place='crafting_table'),
}

AXE_ACTIONS = {
    **BASIC_ACTIONS,
    'craft_planks': dict(craft='planks'),
    'craft_stick': dict(craft='stick'),
    'craft_crafting_table': dict(craft='crafting_table'),
    'place_crafting_table': dict(place='crafting_table'),
    'craft_wooden_axe': dict(nearbyCraft='wooden_axe'),
    'equip_wooden_axe': dict(equip='wooden_axe'),
}

DIAMOND_ACTIONS = {
    **BASIC_ACTIONS,
    'craft_planks': dict(craft='planks'),
    'craft_stick': dict(craft='stick'),
    'craft_torch': dict(craft='torch'),
    'craft_crafting_table': dict(craft='crafting_table'),
    'craft_furnace': dict(nearbyCraft='furnace'),
    'craft_wooden_pickaxe': dict(nearbyCraft='wooden_pickaxe'),
    'craft_stone_pickaxe': dict(nearbyCraft='stone_pickaxe'),
    'craft_iron_pickaxe': dict(nearbyCraft='iron_pickaxe'),
    'smelt_coal': dict(nearbySmelt='coal'),
    'smelt_iron_ingot': dict(nearbySmelt='iron_ingot'),
    'place_torch': dict(place='torch'),
    'place_cobblestone': dict(place='cobblestone'),
    'place_crafting_table': dict(place='crafting_table'),
    'place_furnace': dict(place='furnace'),
    'equip_iron_pickaxe': dict(equip='iron_pickaxe'),
    'equip_stone_pickaxe': dict(equip='stone_pickaxe'),
    'equip_wooden_pickaxe': dict(equip='wooden_pickaxe'),
}

DISCOVER_ACTIONS = {
    **BASIC_ACTIONS,

    'craft_planks': dict(craft='planks'),
    'craft_stick': dict(craft='stick'),
    'craft_torch': dict(craft='torch'),
    'craft_wheat': dict(craft='wheat'),
    'craft_crafting_table': dict(craft='crafting_table'),

    'craft_furnace': dict(nearbyCraft='furnace'),
    'craft_trapdoor': dict(nearbyCraft='trapdoor'),
    'craft_boat': dict(nearbyCraft='boat'),
    'craft_bread': dict(nearbyCraft='bread'),
    'craft_bucket': dict(nearbyCraft='bucket'),
    'craft_ladder': dict(nearbyCraft='ladder'),
    'craft_fence': dict(nearbyCraft='fence'),
    'craft_chest': dict(nearbyCraft='chest'),
    'craft_bowl': dict(nearbyCraft='bowl'),

    'craft_wooden_pickaxe': dict(nearbyCraft='wooden_pickaxe'),
    'craft_wooden_sword': dict(nearbyCraft='wooden_sword'),
    'craft_wooden_shovel': dict(nearbyCraft='wooden_shovel'),
    'craft_wooden_axe': dict(nearbyCraft='wooden_axe'),

    'craft_stone_pickaxe': dict(nearbyCraft='stone_pickaxe'),
    'craft_stone_sword': dict(nearbyCraft='stone_sword'),
    'craft_stone_shovel': dict(nearbyCraft='stone_shovel'),
    'craft_stone_axe': dict(nearbyCraft='stone_axe'),

    'craft_iron_pickaxe': dict(nearbyCraft='iron_pickaxe'),
    'craft_iron_sword': dict(nearbyCraft='iron_sword'),
    'craft_iron_shovel': dict(nearbyCraft='iron_shovel'),
    'craft_iron_axe': dict(nearbyCraft='iron_axe'),

    'smelt_coal': dict(nearbySmelt='coal'),
    'smelt_iron_ingot': dict(nearbySmelt='iron_ingot'),

    'place_torch': dict(place='torch'),
    'place_cobblestone': dict(place='cobblestone'),
    'place_crafting_table': dict(place='crafting_table'),
    'place_dirt': dict(place='dirt'),
    'place_furnace': dict(place='furnace'),

    'equip_iron_pickaxe': dict(equip='iron_pickaxe'),
    'equip_stone_pickaxe': dict(equip='stone_pickaxe'),
    'equip_wooden_pickaxe': dict(equip='wooden_pickaxe'),
}
