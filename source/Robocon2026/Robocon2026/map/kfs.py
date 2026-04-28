import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
import numpy as np

def create_KFS(color):
    kfs = []
    if color == "red":
        coords = np.array(
            [
                [-4.2, 2.2, 0.375],
                [-3.0, 2.2, 0.575],
                [-1.8, 2.2, 0.375],
                [-4.2, 1.0, 0.575],
                [-3.0, 1.0, 0.775],
                [-1.8, 1.0, 0.575],
                [-4.2, -0.2, 0.375],
                [-3.0, -0.2, 0.575],
                [-1.8, -0.2, 0.775],
                [-4.2, -1.4, 0.575],
                [-3.0, -1.4, 0.375],
                [-1.8, -1.4, 0.575],
            ]
        )
    elif color == "blue":
        coords = np.array(
            [
                [4.2, 2.2, 0.375],
                [3.0, 2.2, 0.575],
                [1.8, 2.2, 0.375],
                [4.2, 1.0, 0.575],
                [3.0, 1.0, 0.775],
                [1.8, 1.0, 0.575],
                [4.2, -0.2, 0.375],
                [3.0, -0.2, 0.575],
                [1.8, -0.2, 0.775],
                [4.2, -1.4, 0.575],
                [3.0, -1.4, 0.375],
                [1.8, -1.4, 0.575],
            ]
        )
    # 任选
    # selected_indices = np.random.choice(12, 8, replace=False)
    # r1_indices = selected_indices[:3]
    # r1 排除中间位置
    while True:
        selected_indices = np.random.choice(12, 8, replace=False)
        r1_indices = selected_indices[:3]
        if 4 not in r1_indices and 7 not in r1_indices:
            break
    fake_index = selected_indices[3]
    r2_indices = selected_indices[4:]

    for idx in r1_indices:
        pos = coords[idx]
        kfs.append(
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}" + f"/r1_{color}_{idx}",
                spawn=sim_utils.UsdFileCfg(usd_path=f"assets/KFS/r1_{color}.usd"),
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
            )
        )

    pos = coords[fake_index]
    kfs.append(
        AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}" + f"/fake_{color}",
            spawn=sim_utils.UsdFileCfg(usd_path=f"assets/KFS/fake_{color}.usd"),
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        )
    )

    used_x = set()
    for idx in r2_indices:
        pos = coords[idx]
        while True:
            x = np.random.randint(1, 16)
            if x not in used_x:
                used_x.add(x)
                break
        kfs.append(
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}" + f"/r1_{color}_{idx}",
                spawn=sim_utils.UsdFileCfg(usd_path=f"assets/KFS/r2_{color}_{x}.usd"),
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
            )
        )

    return kfs
