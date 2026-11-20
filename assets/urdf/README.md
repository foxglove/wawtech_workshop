# reBot B601 URDF

`rebot-devarm_description_fixend/` is the reBot-DevArm (B601) model from Seeed Studio's
[reBot-DevArm](https://github.com/Seeed-Projects/reBot-DevArm), under the
[CERN Open Hardware Licence v2 - Weakly Reciprocal](https://github.com/Seeed-Projects/reBot-DevArm/blob/main/LICENSE).

Changes: the Luxonis OAK mount (`oak`, `oak_mount_joint`, ...) from the `foxglove_rebot_luxonis`
demo was removed, and the OAK mesh is not included. The arm has six revolute joints,
`joint1` to `joint6` (the third is spelled `join3` in the model); there is no gripper joint.

`record-episode` serves this folder to Foxglove as `package://` URLs, so the 3D panel loads
`package://rebot-devarm_description_fixend/urdf/reBot-DevArm_fixend.urdf`.

Mesh paths in the URDF are relative (`../meshes/link1.STL`) rather than `package://`, and the package folder
name is lowercase: Foxglove lower-cases `package://` names and could not find a mixed-case folder, and relative
paths load next to the URDF whether it is read from disk or fetched from the recorder.
