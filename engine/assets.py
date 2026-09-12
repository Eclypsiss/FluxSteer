"""五个数据集在 MAIN 评估中使用的对象资产路径。"""

from pathlib import Path


def object_asset_candidates(dataset_dir, object_name):
    root = Path(dataset_dir)
    if "+" in object_name:
        category, name = object_name.split("+", 1)
        directory = root / "object" / category / name
        return [directory / f"{name}.urdf"], [directory / f"{name}.stl"]
    directories = [root / name for name in ("contact_meshes", "meshdata", "mesh", "obj_scale_urdf")]
    urdfs = [directory / f"{object_name}.urdf" for directory in directories]
    meshes = [directory / f"{object_name}.{suffix}" for directory in directories
              for suffix in ("ply", "obj", "stl")]
    return urdfs, meshes


def find_object_assets(dataset_dir, object_name):
    urdfs, meshes = object_asset_candidates(dataset_dir, object_name)
    urdf = next((p for p in urdfs if p.is_file()), None)
    mesh = next((p for p in meshes if p.is_file()), None)
    if urdf is None or mesh is None:
        raise FileNotFoundError(f"对象 {object_name} 缺少 MAIN 所需的平铺 URDF 或 mesh：{dataset_dir}")
    return str(urdf.resolve()), str(mesh.resolve())


def normals_cache(dataset_dir):
    root = Path(dataset_dir)
    for name in ("scaled_object_pcds_nors.pkl", "object_pcds_nors.pkl"):
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"缺少 collision 点云法向缓存：{dataset_dir}")
