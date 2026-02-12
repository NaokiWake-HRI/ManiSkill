"""Panda arm + Allegro Hand Right with FSR tactile sensors."""
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
class PandaAllegroTouch(PandaAllegro):
    uid = "panda_allegro_touch"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_allegro_fsr.urdf"

    # FSR link names (with allegro_ prefix matching the URDF)
    # Order: thumb (root→tip), index (root→tip), middle (root→tip), ring (root→tip)
    finger_fsr_link_names = [
        "allegro_link_14.0_fsr",
        "allegro_link_15.0_fsr",
        "allegro_link_15.0_tip_fsr",
        "allegro_link_1.0_fsr",
        "allegro_link_2.0_fsr",
        "allegro_link_3.0_tip_fsr",
        "allegro_link_5.0_fsr",
        "allegro_link_6.0_fsr",
        "allegro_link_7.0_tip_fsr",
        "allegro_link_9.0_fsr",
        "allegro_link_10.0_fsr",
        "allegro_link_11.0_tip_fsr",
    ]
    palm_fsr_link_names = [
        "allegro_link_base_fsr",
        "allegro_link_0.0_fsr",
        "allegro_link_4.0_fsr",
        "allegro_link_8.0_fsr",
    ]

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
        self.fsr_links: list[Actor] = sapien_utils.get_objs_by_names(
            self.robot.get_links(),
            self.palm_fsr_link_names + self.finger_fsr_link_names,
        )

    def get_fsr_obj_impulse(self, obj: Actor = None):
        """Get contact impulse between each FSR link and a specific object.

        Returns shape: GPU: (n_fsr, n_envs, 3), CPU: (n_fsr, 3)
        """
        if self.scene.gpu_sim_enabled:
            px: physx.PhysxGpuSystem = self.scene.px
            if obj.name not in self.pair_query:
                bodies = list(zip(*[link._bodies for link in self.fsr_links]))
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
                    (len(obj._bodies), len(self.fsr_links), 3),
                )
            query, contacts_shape = self.pair_query[obj.name]
            px.gpu_query_contact_pair_impulses(query)
            contacts = (
                query.cuda_impulses.torch()
                .clone()
                .reshape((len(self.fsr_links), *contacts_shape))
            )
            return contacts
        else:
            internal_fsr_links = [
                link._bodies[0].entity for link in self.fsr_links
            ]
            contacts = self.scene.get_contacts()
            obj_contacts = sapien_utils.get_multiple_pairwise_contacts(
                contacts, obj._bodies[0].entity, internal_fsr_links
            )
            sorted_contacts = [obj_contacts[link] for link in internal_fsr_links]
            contact_forces = [
                sapien_utils.compute_total_impulse(contact)
                for contact in sorted_contacts
            ]
            return np.stack(contact_forces)

    def get_fsr_impulse(self):
        """Get contact impulse for each FSR link against all objects.

        Returns shape: GPU: (n_envs, n_fsr, 3), CPU: (1, n_fsr, 3)
        """
        if self.scene.gpu_sim_enabled:
            px: physx.PhysxGpuSystem = self.scene.px
            if self.body_query is None:
                bodies = list(
                    zip(*[link._bodies for link in self.fsr_links])
                )
                bodies = list(itertools.chain(*bodies))
                query = px.gpu_create_contact_body_impulse_query(bodies)
                self.body_query = (
                    query,
                    (
                        len(self.fsr_links[0]._bodies),
                        len(self.fsr_links),
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
            internal_fsr_links = [
                link._bodies[0].entity for link in self.fsr_links
            ]
            contacts = self.scene.get_contacts()
            contact_map = sapien_utils.get_cpu_actors_contacts(
                contacts, internal_fsr_links
            )
            sorted_contacts = [contact_map[link] for link in internal_fsr_links]
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
        fsr_impulse = self.get_fsr_impulse()
        # Add FSR force magnitudes (16 scalars) to proprioception
        obs.update({"fsr_impulse": torch.linalg.norm(fsr_impulse, dim=-1)})
        return obs
