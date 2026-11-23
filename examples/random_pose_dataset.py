"""Initialise Open G1 in N random joint configurations and record everything MuJoCo computes for each.

Each sample:
  * every hinge is drawn uniformly from its full joint range (the clipped servo travel for the arms)
  * the base stays at the model's default pose (x = y = 0, pelvis z = 0.793, upright) unless
    --random-yaw is given; the robot is NOT dropped or settled, it is a pure kinematic snapshot
  * joint velocities are 0 unless --vel-scale > 0 (then uniform in +/- vel-scale rad/s, base included)
  * mj_forward is run, then positions, velocities, the env observation, body / site poses, centre of mass,
    sensors and contacts are recorded

Outputs (in --out-dir):
  random_poses.npz        every array, shape (N, ...), plus name lists so you can index by name
  random_poses.csv        one row per sample, one labelled column per scalar (easy to open / pandas)
  joint_static.json       the per-joint info that doesn't change between samples (indices, masses, ranges, ...)

Usage:
  python tools/random_pose_dataset.py                      # 1000 samples, seed 0
  python tools/random_pose_dataset.py -n 5000 --seed 3 --vel-scale 2.0 --random-yaw
"""
import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XML = ROOT / "src" / "openG1" / "assets" / "open_g1.xml"
OBS_QPOS_DROP = 2                       # OpenG1StandEnv drops world x, y from qpos
SITES = ["left_hand", "right_hand", "imu_in_pelvis", "imu_in_torso"]
FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]


# ---- per-joint static info (self-contained copy of tools/joint_info.py) ----
JOINT_TYPES = {int(mujoco.mjtJoint.mjJNT_FREE): "free", int(mujoco.mjtJoint.mjJNT_BALL): "ball",
               int(mujoco.mjtJoint.mjJNT_SLIDE): "slide", int(mujoco.mjtJoint.mjJNT_HINGE): "hinge"}
QPOS_WIDTH = {"free": 7, "ball": 4, "slide": 1, "hinge": 1}
QVEL_WIDTH = {"free": 6, "ball": 3, "slide": 1, "hinge": 1}
FREE_QPOS_LABELS = ["x", "y", "z", "quat_w", "quat_x", "quat_y", "quat_z"]
FREE_QVEL_LABELS = ["vx", "vy", "vz", "wx", "wy", "wz"]


def _actuators_for_joint(model, jid):
    """Actuators that drive this joint, directly or through a tendon."""
    out = []
    for a in range(model.nu):
        trn, target = model.actuator_trntype[a], model.actuator_trnid[a, 0]
        if trn == mujoco.mjtTrn.mjTRN_JOINT and target == jid:
            out.append({"index": a, "name": model.actuator(a).name, "via": "direct",
                        "ctrlrange": model.actuator_ctrlrange[a].tolist()})
        elif trn == mujoco.mjtTrn.mjTRN_TENDON:
            adr, num = model.tendon_adr[target], model.tendon_num[target]
            for w in range(adr, adr + num):
                if model.wrap_type[w] == mujoco.mjtWrap.mjWRAP_JOINT and model.wrap_objid[w] == jid:
                    out.append({"index": a, "name": model.actuator(a).name,
                                "via": f"tendon {model.tendon(target).name} (coef {model.wrap_prm[w]:+.4f})",
                                "ctrlrange": model.actuator_ctrlrange[a].tolist()})
    return out


def joint_info(model, data=None, obs_qpos_drop=None):
    """Return {joint_name: {...}} in qpos order.

    obs_qpos_drop: how many leading qpos entries the env's observation drops before concatenating qvel
                   (2 for OpenG1StandEnv, which drops world x/y; 0 for the upper-body env).
                   None leaves the obs indices out.
    """
    info = {}
    for j in range(model.njnt):
        name = model.joint(j).name
        jtype = JOINT_TYPES[int(model.jnt_type[j])]
        qadr, vadr = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
        qw, vw = QPOS_WIDTH[jtype], QVEL_WIDTH[jtype]
        body = int(model.jnt_bodyid[j])

        entry = {
            "type": jtype,
            "body": model.body(body).name,
            "qpos_index": qadr if qw == 1 else list(range(qadr, qadr + qw)),
            "qvel_index": vadr if vw == 1 else list(range(vadr, vadr + vw)),
            "link_mass": float(model.body_mass[body]),
            "subtree_mass": float(model.body_subtreemass[body]),
            "range": model.jnt_range[j].tolist() if model.jnt_limited[j] else None,
            "armature": float(model.dof_armature[vadr]),
            "damping": float(model.dof_damping[vadr]),
            "frictionloss": float(model.dof_frictionloss[vadr]),
            "actuators": _actuators_for_joint(model, j),
        }
        if jtype == "free":
            entry["qpos_labels"] = FREE_QPOS_LABELS
            entry["qvel_labels"] = FREE_QVEL_LABELS

        if obs_qpos_drop is not None:
            vel_offset = model.nq - obs_qpos_drop
            pos_idx = [i - obs_qpos_drop for i in range(qadr, qadr + qw) if i >= obs_qpos_drop]
            vel_idx = [vel_offset + i for i in range(vadr, vadr + vw)]
            entry["obs_pos_index"] = pos_idx[0] if qw == 1 else pos_idx
            entry["obs_vel_index"] = vel_idx[0] if vw == 1 else vel_idx

        if data is not None:
            pos, vel = data.qpos[qadr:qadr + qw], data.qvel[vadr:vadr + vw]
            entry["position"] = float(pos[0]) if qw == 1 else pos.tolist()
            entry["velocity"] = float(vel[0]) if vw == 1 else vel.tolist()

        info[name] = entry
    return info


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def sample_pose(model, rng, vel_scale, random_yaw):
    qpos, qvel = model.qpos0.copy(), np.zeros(model.nv)
    for j in range(model.njnt):
        a, d = model.jnt_qposadr[j], model.jnt_dofadr[j]
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            if random_yaw:
                qpos[a + 3:a + 7] = yaw_quat(rng.uniform(-np.pi, np.pi))
            if vel_scale > 0:
                qvel[d:d + 6] = rng.uniform(-vel_scale, vel_scale, 6)
        else:                                       # hinge / slide
            lo, hi = model.jnt_range[j] if model.jnt_limited[j] else (-np.pi, np.pi)
            qpos[a] = rng.uniform(lo, hi)
            if vel_scale > 0:
                qvel[d] = rng.uniform(-vel_scale, vel_scale)
    return qpos, qvel


def contact_stats(model, data):
    n_self = n_floor = 0
    min_dist = np.inf
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if b1 == 0 or b2 == 0:
            n_floor += 1
        else:
            n_self += 1
        min_dist = min(min_dist, c.dist)
    return data.ncon, n_self, n_floor, (min_dist if data.ncon else 0.0)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-n", "--num", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--xml", default=str(DEFAULT_XML))
    p.add_argument("--vel-scale", type=float, default=0.0, help="random joint/base velocity bound (rad/s, m/s)")
    p.add_argument("--random-yaw", action="store_true", help="randomise the base heading")
    p.add_argument("--out-dir", default="data")
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(args.seed)
    N = args.num

    joint_names = [model.joint(j).name for j in range(model.njnt)]
    hinge_ids = [j for j in range(model.njnt) if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    hinge_names = [joint_names[j] for j in hinge_ids]
    body_names = [model.body(b).name for b in range(model.nbody)]
    sensor_names = [model.sensor(s).name for s in range(model.nsensor)]
    site_ids = [model.site(s).id for s in SITES]
    foot_ids = [model.body(b).id for b in FEET]

    rec = {
        "qpos": np.zeros((N, model.nq)),
        "qvel": np.zeros((N, model.nv)),
        "obs": np.zeros((N, model.nq - OBS_QPOS_DROP + model.nv)),
        "joint_pos": np.zeros((N, len(hinge_ids))),
        "joint_vel": np.zeros((N, len(hinge_ids))),
        "body_xpos": np.zeros((N, model.nbody, 3)),
        "body_xquat": np.zeros((N, model.nbody, 4)),
        "site_xpos": np.zeros((N, len(SITES), 3)),
        "com": np.zeros((N, 3)),
        "sensordata": np.zeros((N, model.nsensordata)),
        "upright": np.zeros(N),               # torso z-axis . world up (same as the env reward term)
        "foot_z": np.zeros((N, 2)),
        "lowest_geom_z": np.zeros(N),         # < 0 means part of the robot is below the floor
        "ncon": np.zeros(N, dtype=int),
        "n_self_contacts": np.zeros(N, dtype=int),
        "n_floor_contacts": np.zeros(N, dtype=int),
        "min_contact_dist": np.zeros(N),      # most negative = deepest penetration
    }
    torso = model.body("torso_link").id
    hinge_qadr = model.jnt_qposadr[hinge_ids]
    hinge_dadr = model.jnt_dofadr[hinge_ids]
    collide_geoms = np.where((model.geom_contype | model.geom_conaffinity) & (model.geom_bodyid > 0))[0]

    for i in range(N):
        qpos, qvel = sample_pose(model, rng, args.vel_scale, args.random_yaw)
        mujoco.mj_resetData(model, data)
        data.qpos[:], data.qvel[:] = qpos, qvel
        mujoco.mj_forward(model, data)

        rec["qpos"][i], rec["qvel"][i] = data.qpos, data.qvel
        rec["obs"][i] = np.concatenate([data.qpos[OBS_QPOS_DROP:], data.qvel])
        rec["joint_pos"][i], rec["joint_vel"][i] = data.qpos[hinge_qadr], data.qvel[hinge_dadr]
        rec["body_xpos"][i], rec["body_xquat"][i] = data.xpos, data.xquat
        rec["site_xpos"][i] = data.site_xpos[site_ids]
        rec["com"][i] = data.subtree_com[1]                   # body 1 = pelvis = whole robot
        rec["sensordata"][i] = data.sensordata
        rec["upright"][i] = data.xmat[torso].reshape(3, 3)[2, 2]
        rec["foot_z"][i] = data.xpos[foot_ids, 2]
        rec["lowest_geom_z"][i] = np.min(data.geom_xpos[collide_geoms, 2] - model.geom_rbound[collide_geoms])
        (rec["ncon"][i], rec["n_self_contacts"][i],
         rec["n_floor_contacts"][i], rec["min_contact_dist"][i]) = contact_stats(model, data)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out / "random_poses.npz", **rec,
                        joint_names=np.array(hinge_names), body_names=np.array(body_names),
                        site_names=np.array(SITES), sensor_names=np.array(sensor_names),
                        seed=args.seed, vel_scale=args.vel_scale, random_yaw=args.random_yaw)

    # flat CSV: one labelled column per scalar
    cols, blocks = ["sample"], [np.arange(N)[:, None]]
    def add(prefix_names, arr):
        cols.extend(prefix_names); blocks.append(arr.reshape(N, -1))
    add(["base_x", "base_y", "base_z", "base_qw", "base_qx", "base_qy", "base_qz"], rec["qpos"][:, :7])
    add(["base_vx", "base_vy", "base_vz", "base_wx", "base_wy", "base_wz"], rec["qvel"][:, :6])
    add([f"{n}_pos" for n in hinge_names], rec["joint_pos"])
    add([f"{n}_vel" for n in hinge_names], rec["joint_vel"])
    add([f"{s}_{a}" for s in SITES for a in "xyz"], rec["site_xpos"])
    add(["com_x", "com_y", "com_z"], rec["com"])
    sensor_cols = []
    for s in range(model.nsensor):
        dim = model.sensor_dim[s]
        sensor_cols += [f"{sensor_names[s]}_{k}" for k in ("xyz" if dim == 3 else range(dim))]
    add(sensor_cols, rec["sensordata"])
    add(["upright", "left_foot_z", "right_foot_z", "lowest_geom_z"],
        np.column_stack([rec["upright"], rec["foot_z"], rec["lowest_geom_z"]]))
    add(["ncon", "n_self_contacts", "n_floor_contacts", "min_contact_dist"],
        np.column_stack([rec["ncon"], rec["n_self_contacts"], rec["n_floor_contacts"], rec["min_contact_dist"]]))
    table = np.hstack(blocks)
    with open(out / "random_poses.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in table:
            w.writerow([f"{v:.6g}" for v in row])

    (out / "joint_static.json").write_text(json.dumps(joint_info(model, obs_qpos_drop=OBS_QPOS_DROP), indent=2))

    print(f"{N} samples -> {out}/random_poses.npz, random_poses.csv ({len(cols)} columns), joint_static.json")
    print(f"  with self-collision : {np.mean(rec['n_self_contacts'] > 0):6.1%}")
    print(f"  touching the floor  : {np.mean(rec['n_floor_contacts'] > 0):6.1%}")
    print(f"  any geom below floor: {np.mean(rec['lowest_geom_z'] < 0):6.1%}")


if __name__ == "__main__":
    main()