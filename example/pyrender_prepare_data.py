"""Examples of using pyrender for viewing and offscreen rendering.
"""

import numpy as np
import trimesh
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, field_validator, Field

from pyrender import (
    PerspectiveCamera,
    DirectionalLight,
    SpotLight,
    PointLight,
    Mesh,
    Node,
    Scene,
    OffscreenRenderer,
)
import cv2


class OrbitZCamRig:
    """Simple camera rig that orbits around the Z axis at a fixed height and radius.
    default pose:
    np.array([
        [0.0, -np.sqrt(2)/2, np.sqrt(2)/2, 0.5],
        [1.0,  0.0,          0.0,           0.0],
        [0.0,  np.sqrt(2)/2, np.sqrt(2)/2,  0.5],
        [0.0,  0.0,          0.0,           1.0],
    ])
    """

    def __init__(
        self,
        z_height=0.5,
        radius=0.5,
        look_at_target: Optional[tuple[float, float, float]] = None,
    ):
        self.z_height = z_height
        self.radius = radius
        self.lookAtTarget = np.ndarray(look_at_target) if look_at_target is not None else np.array([0.0, 0.0, 0.0])

    def get_camera_pose(self, angle_rad: float = 0.0) -> np.ndarray:
        cam_x = self.radius * np.cos(angle_rad)
        cam_y = self.radius * np.sin(angle_rad)
        cam_z = self.z_height

        forward = self.lookAtTarget - np.array([cam_x, cam_y, cam_z])
        forward /= np.linalg.norm(forward)

        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)  # Bug 1 fix: forward × up, not up × forward
        right /= np.linalg.norm(right)

        up = np.cross(right, forward)  # Bug 2 fix: right × forward, not forward × right

        pose = np.eye(4)
        pose[0:3, 0] = right
        pose[0:3, 1] = up
        pose[0:3, 2] = -forward  # Bug 3 fix: OpenGL camera looks down -Z
        pose[0:3, 3] = [cam_x, cam_y, cam_z]

        return pose


class RenderCamera(BaseModel):
    """Camera config for pyrender scenes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    yfov: float = Field(default=np.pi / 3.0)
    screen_size: Tuple[int, int] = Field(
        default=(640 * 2, 480 * 2),
        description="Width and height of the rendered image in pixels.",
    )

    # 4x4 camera-to-world matrix (OpenGL format)
    pose: np.typing.NDArray[np.float64] = Field(default_factory=lambda: OrbitZCamRig().get_camera_pose())

    @field_validator("pose")
    @classmethod
    def validate_pose(cls, v):
        v = np.array(v, dtype=float)
        if v.shape != (4, 4):
            raise ValueError(f"pose must be a 4x4 matrix, got shape {v.shape}")
        return v

    @property
    def camera(self) -> PerspectiveCamera:
        return PerspectiveCamera(yfov=self.yfov)


class OCVCameraIntrinsics(BaseModel):
    """Camera config for OpenCV-based rendering."""

    focal_length: Tuple[float, float] = Field(default=(800.0, 800.0))
    principal_point: Tuple[float, float] = Field(default=(320.0, 240.0))
    distortion_coeffs: Tuple[float, float, float, float, float] = Field(
        default=(0.0, 0.0, 0.0, 0.0, 0.0),
        description="Distortion coefficients (k1, k2, p1, p2, k3) for the camera.",
    )


class RenderScene:
    def __init__(self, camera_obj: RenderCamera):
        self.scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))
        self._camera = camera_obj
        self._build_meshes()
        self._build_lights()
        self._add_nodes()

    def _build_meshes(self):
        fuze_trimesh = trimesh.load("./models/fuze.obj")
        self._fuze_mesh = Mesh.from_trimesh(fuze_trimesh)
        self._fuze_z_offset = -np.min(fuze_trimesh.vertices[:, 2])

        drill_trimesh = trimesh.load("./models/drill.obj")
        self._drill_mesh = Mesh.from_trimesh(drill_trimesh)
        self._drill_pose = np.eye(4)
        self._drill_pose[0, 3] = 0.1
        self._drill_pose[2, 3] = -np.min(drill_trimesh.vertices[:, 2])

        wood_trimesh = trimesh.load("./models/wood.obj")
        self._wood_mesh = Mesh.from_trimesh(wood_trimesh)

        bottle_gltf = trimesh.load("./models/WaterBottle.glb")
        bottle_trimesh = bottle_gltf.geometry[list(bottle_gltf.geometry.keys())[0]]
        self._bottle_mesh = Mesh.from_trimesh(bottle_trimesh)
        self._bottle_pose = np.array(
            [
                [1.0, 0.0, 0.0, 0.1],
                [0.0, 0.0, -1.0, -0.16],
                [0.0, 1.0, 0.0, 0.13],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        points = trimesh.creation.icosphere(radius=0.05).vertices
        self._points_mesh = Mesh.from_points(points, colors=np.random.uniform(size=points.shape))

    def _build_lights(self):
        self._direc_l = DirectionalLight(color=np.ones(3), intensity=1.0)
        self._spot_l = SpotLight(
            color=np.ones(3),
            intensity=10.0,
            innerConeAngle=np.pi / 16,
            outerConeAngle=np.pi / 6,
        )
        self._point_l = PointLight(color=np.ones(3), intensity=10.0)

    def _add_nodes(self):
        cam_pose = self._camera.pose

        self.scene.add_node(
            Node(
                mesh=self._fuze_mesh,
                translation=np.array([0.1, 0.15, self._fuze_z_offset]),
            )
        )

        self.drill_node = self.scene.add(self._drill_mesh, pose=self._drill_pose)
        self.scene.add(self._bottle_mesh, pose=self._bottle_pose)
        self.scene.add(self._wood_mesh)
        self.scene.add(self._direc_l, pose=cam_pose)
        self.scene.add(self._spot_l, pose=cam_pose)

        self.cam_node = self.scene.add(self._camera.camera, pose=cam_pose)

    @property
    def camera(self):
        return self._camera

    @camera.setter
    def camera(self, camera_obj: RenderCamera):
        self._camera = camera_obj
        self.scene.set_pose(self.cam_node, camera_obj.pose)

    def export_pointcloud(self, output_path, max_points_per_primitive=None):
        all_points = []

        for node in self.scene.mesh_nodes:
            pose = self.scene.get_pose(node)
            for primitive in node.mesh.primitives:
                positions = primitive.positions  # (N, 3)

                if max_points_per_primitive and len(positions) > max_points_per_primitive:
                    idx = np.random.choice(len(positions), max_points_per_primitive, replace=False)
                    positions = positions[idx]
                else:
                    idx = None

                ones = np.ones((len(positions), 1))
                pts_h = np.hstack([positions, ones])
                world_pts = (pose @ pts_h.T).T[:, :3]
                all_points.append(world_pts)

        all_points = np.vstack(all_points)
        trimesh.PointCloud(vertices=all_points).export(output_path)


if __name__ == "__main__":
    cam_rig = OrbitZCamRig()
    cam_pose_list = [cam_rig.get_camera_pose(angle) for angle in np.linspace(0, 2 * np.pi, num=16, endpoint=False)]

    cameras = [RenderCamera(pose=pose) for pose in cam_pose_list]

    render_scene = RenderScene(cameras[0])
    render_scene.export_pointcloud("data/pyrender_scene.ply", max_points_per_primitive=500)

    for idx, camera in enumerate(cameras):
        render_scene.camera = camera
        r = OffscreenRenderer(viewport_width=camera.screen_size[0], viewport_height=camera.screen_size[1])
        color, depth = r.render(render_scene.scene)
        cv2.imwrite(f"data/pyrender_scene_{idx}.png", cv2.cvtColor(color, cv2.COLOR_RGBA2BGRA))

    r.delete()
