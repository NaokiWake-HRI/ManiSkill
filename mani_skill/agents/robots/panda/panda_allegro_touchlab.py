"""Panda arm + Allegro Hand Right with TouchLab tactile sensor patches."""
import itertools
from typing import Optional, Tuple

import numpy as np
import sapien
import torch
from sapien import physx

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.panda.panda_allegro import PandaAllegro
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.actor import Actor


@register_agent()
class PandaAllegroTouchLab(PandaAllegro):
    uid = "panda_allegro_touchlab"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_allegro_touchlab.urdf"

    # TouchLab sensor patch link names (4 tip sensors)
    finger_tl_link_names = [
        "allegro_link_3.0_tip_tl",   # Index tip
        "allegro_link_7.0_tip_tl",   # Middle tip
        "allegro_link_11.0_tip_tl",  # Ring tip
        "allegro_link_15.0_tip_tl",  # Thumb tip
    ]
    palm_tl_link_names = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pair_query: dict[
            str, Tuple[physx.PhysxGpuContactPairImpulseQuery, Tuple[int, int, int]]
        ] = dict()
        self.body_query: Optional[
            Tuple[physx.PhysxGpuContactBodyImpulseQuery, Tuple[int, int, int]]
        ] = None

    def _after_init(self):
        super()._after_init()
        self.tl_links: list[Actor] = sapien_utils.get_objs_by_names(
            self.robot.get_links(),
            self.palm_tl_link_names + self.finger_tl_link_names,
        )

    def get_tl_obj_impulse(self, obj: Actor = None):
        """Get contact impulse between each TouchLab patch and a specific object.

        Returns shape: GPU: (n_tl, n_envs, 3), CPU: (n_tl, 3)
        """
        if self.scene.gpu_sim_enabled:
            px: physx.PhysxGpuSystem = self.scene.px
            if obj.name not in self.pair_query:
                bodies = list(zip(*[link._bodies for link in self.tl_links]))
                bodies = list(itertools.chain(*bodies))
                obj_bodies = [
                    elem
                    for item in obj._bodies
                    for elem in itertools.repeat(item, 2)
                ]
                body_pairs = list(zip(bodies, obj_bodies))
                query = px.gpu_create_contact_pair_impulse_query(body_pairs)
                self.pair_query[obj.name] = (
                    query,
                    (len(obj._bodies), len(self.tl_links), 3),
                )
            query, contacts_shape = self.pair_query[obj.name]
            px.gpu_query_contact_pair_impulses(query)
            contacts = (
                query.cuda_impulses.torch()
                .clone()
                .reshape((len(self.tl_links), *contacts_shape))
            )
            return contacts
        else:
            internal_tl_links = [
                link._bodies[0].entity for link in self.tl_links
            ]
            contacts = self.scene.get_contacts()
            obj_contacts = sapien_utils.get_multiple_pairwise_contacts(
                contacts, obj._bodies[0].entity, internal_tl_links
            )
            sorted_contacts = [obj_contacts[link] for link in internal_tl_links]
            contact_forces = [
                sapien_utils.compute_total_impulse(contact)
                for contact in sorted_contacts
            ]
            return np.stack(contact_forces)

    def get_tl_impulse(self):
        """Get contact impulse for each TouchLab patch against all objects.

        Returns shape: GPU: (n_envs, n_tl, 3), CPU: (1, n_tl, 3)
        """
        if self.scene.gpu_sim_enabled:
            px: physx.PhysxGpuSystem = self.scene.px
            if self.body_query is None:
                bodies = list(
                    zip(*[link._bodies for link in self.tl_links])
                )
                bodies = list(itertools.chain(*bodies))
                query = px.gpu_create_contact_body_impulse_query(bodies)
                self.body_query = (
                    query,
                    (
                        len(self.tl_links[0]._bodies),
                        len(self.tl_links),
                        3,
                    ),
                )
            query, contacts_shape = self.body_query
            px.gpu_query_contact_body_impulses(query)
            contacts = (
                query.cuda_impulses.torch().clone().reshape(*contacts_shape)
            )
            return contacts
        else:
            internal_tl_links = [
                link._bodies[0].entity for link in self.tl_links
            ]
            contacts = self.scene.get_contacts()
            contact_map = sapien_utils.get_cpu_actors_contacts(
                contacts, internal_tl_links
            )
            sorted_contacts = [contact_map[link] for link in internal_tl_links]
            contact_forces = [
                sapien_utils.compute_total_impulse(contact)
                for contact in sorted_contacts
            ]
            contact_impulse = torch.from_numpy(
                np.stack(contact_forces)[None, ...]
            )
            return contact_impulse

    def get_proprioception(self):
        obs = super().get_proprioception()
        tl_impulse = self.get_tl_impulse()
        obs.update({"tl_impulse": torch.linalg.norm(tl_impulse, dim=-1)})
        return obs
