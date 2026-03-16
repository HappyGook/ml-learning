from manim import *


class GradientScene(ThreeDScene):

    def __init__(self, func, classic_path, momentum_path, **kwargs):
        super().__init__(**kwargs)
        self.func = func
        self.classic_path = classic_path
        self.momentum_path = momentum_path

    def construct(self):

        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[0, 10, 2],
        )
        self.add(axes)

        surface = self.create_surface(axes, self.func)
        surface.set_style(fill_opacity=0.6, stroke_color=GRAY)
        self.add(surface)

        # Take the min point to aim the camera towards
        xm, ym = self.classic_path[-1]
        zm = self.func([xm, ym])
        min_point = axes.c2p(xm,ym,zm)

        self.set_camera_orientation(phi=75 * DEGREES, theta=-60 * DEGREES)

        self.move_camera(frame_center=min_point)


        ball_gd = Sphere(radius=0.07)
        ball_gd.set_color(BLUE)

        ball_momentum = Sphere(radius=0.07)
        ball_momentum.set_color(RED)

        x0, y0 = self.classic_path[0]
        ball_gd.move_to(axes.c2p(x0, y0, self.func([x0, y0])))

        x1, y1 = self.momentum_path[0]
        ball_momentum.move_to(axes.c2p(x1, y1, self.func([x1, y1])))

        trail_gd = TracedPath(
            ball_gd.get_center,
            stroke_color=BLUE,
            stroke_width=2,
            stroke_opacity=0.8,
        )
        trail_momentum = TracedPath(
            ball_momentum.get_center,
            stroke_color=RED,
            stroke_width=2,
            stroke_opacity=0.8,
        )

        self.add(trail_gd, trail_momentum)
        self.add(ball_gd, ball_momentum)

        self.animate_paths(axes, ball_gd, ball_momentum)

    def create_surface(self, axes, func):
        return Surface(
            lambda u, v: axes.c2p(u, v, func([u, v])),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(40, 40),
        )

    def animate_paths(self, axes, ball_gd, ball_momentum):
        for p1, p2 in zip(self.classic_path, self.momentum_path):
            x1, y1 = p1
            x2, y2 = p2

            z1 = self.func([x1, y1])
            z2 = self.func([x2, y2])

            self.play(
                ball_gd.animate.move_to(axes.c2p(x1, y1, self.func([x1, y1]))),
                ball_momentum.animate.move_to(axes.c2p(x2, y2, self.func([x2, y2]))),
                run_time=0.2,
            )