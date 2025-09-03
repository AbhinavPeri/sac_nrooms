from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.record import CSVLogger, VideoRecorder

from sac.nrooms import make_env

def make_wrapped_env():
    return GymWrapper(make_env(render_mode=None))


if __name__ == '__main__':
    env = GymWrapper(make_env(render_mode='rgb_array'))

    logger = CSVLogger(
        exp_name="nrooms",
        log_dir="videos_out",  # folder created if missing
        video_format="mp4",  # mp4, avi, gif… CSVLogger supports several
        video_fps=12,  # default FPS; can override in VideoRecorder(fps=..)
    )

    env = TransformedEnv(
        env,
        VideoRecorder(
            logger=logger,
            tag="rollout",  # file tag/prefix
            in_keys=["image"],  # frames to record
            fps=12,  # optional override of logger fps
            skip=1,  # keep every frame; set 2 to halve fps
            make_grid=False,  # True if you feed BxHxWxC batches
        ),
    )

    td = env.rollout(max_steps=512, break_when_any_done=True)
    print(td)
    env.transform.dump()


