"""Collect per-joint info for an Open G1 MuJoCo model into a dictionary.

Joints have no mass of their own in MuJoCo, so each entry reports the mass of the link the joint drives
("link_mass") and of everything downstream of it ("subtree_mass", the load the joint moves).
If you pass an MjData (e.g. env.unwrapped.data) you also get the live position and velocity.

Usage:
    python tools/joint_info.py                           # full robot, stand-env obs layout
    python tools/joint_info.py --xml src/openG1/assets/open_g1_upper.xml --obs-drop 0

    from tools.joint_info import joint_info
    info = joint_info(env.unwrapped.model, env.unwrapped.data, obs_qpos_drop=2)
    info["left_knee_joint"]["velocity"]
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

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


def main():
    default_xml = Path(__file__).resolve().parent.parent / "src" / "openG1" / "assets" / "open_g1.xml"
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--xml", default=str(default_xml))
    p.add_argument("--obs-drop", type=int, default=2, help="2 for the stand env, 0 for the upper-body env")
    p.add_argument("--out", help="write the dictionary to this JSON file")
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    info = joint_info(model, data, obs_qpos_drop=args.obs_drop)

    print(f"{'joint':28s} {'qpos':>7s} {'qvel':>7s} {'obs_p':>7s} {'obs_v':>7s} {'link kg':>8s} {'moved kg':>9s}")
    for name, e in info.items():
        fmt = lambda v: f"{v[0]}-{v[-1]}" if isinstance(v, list) else str(v)
        print(f"{name:28s} {fmt(e['qpos_index']):>7s} {fmt(e['qvel_index']):>7s} "
              f"{fmt(e.get('obs_pos_index', '-')):>7s} {fmt(e.get('obs_vel_index', '-')):>7s} "
              f"{e['link_mass']:8.4f} {e['subtree_mass']:9.3f}")
    print(f"total mass {mujoco.mj_getTotalmass(model):.3f} kg")

    if args.out:
        Path(args.out).write_text(json.dumps(info, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()