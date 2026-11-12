# lerobot_camera_orbbec

LeRobot 0.7 camera plugin. Importing the package registers the `orbbec` camera type.

Stream selection, pixel formats, and device enumeration follow the OrbbecSDK v2 Python wrapper (`pyorbbecsdk`). Behaviour on a real Gemini 305, including macOS, is **unverified on hardware**. The SDK is imported only when a camera connects or when devices are enumerated, so the package imports on machines that do not have a camera attached.

## Usage

```bash
--robot.cameras='{ front: {type: orbbec, width: 1280, height: 720, fps: 30} }'
```

`fps`, `width`, and `height` must be set together or left unset. When they are set, the camera requests that color profile and raises `ValueError` listing the profiles the device reported if it is not available.

Frames returned to LeRobot are RGB uint8 arrays of shape `(height, width, 3)`. The `--robot.cameras` key (`front` in the example) becomes the image topic and feature name, so keep that key equal to `<name>` in `/observation/images/<name>`.
