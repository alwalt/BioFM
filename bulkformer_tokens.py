from manim import *

class TokenBroadcasting(MovingCameraScene):
    def construct(self):

        # --- Title ---
        title = Text("Broadcasting & Summation of Embeddings", font_size=36)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        # Zoom out the camera so everything fits
        self.play(self.camera.frame.animate.scale(1.25))

        # --- Dummy vectors ---
        ree_vec      = Matrix([[1],[0.4],[0.8],[0.2],[0.6]])
        gene_vec     = Matrix([[0.3],[0.1],[0.2],[0.4],[0.9]])
        sample_vec   = Matrix([[0.5],[0.5],[0.5],[0.5],[0.5]])

        ree_label    = Text("REE(expr)", color=BLUE, font_size=24)
        gene_label   = Text("Gene Identity", color=GREEN, font_size=24)
        sample_label = Text("Sample Context", color=ORANGE, font_size=24)

        ree_group = VGroup(ree_label, ree_vec).arrange(DOWN)
        gene_group = VGroup(gene_label, gene_vec).arrange(DOWN)
        sample_group = VGroup(sample_label, sample_vec).arrange(DOWN)

        top_row = VGroup(
            ree_group,
            gene_group,
            sample_group
        ).arrange(RIGHT, buff=2).shift(UP * 0.5)

        self.play(FadeIn(top_row))
        self.wait(1)

        # --- Broadcasting annotation ---
        brace = Brace(sample_vec, direction=RIGHT)

        # FIX: no brace.get_text, use Text() instead
        broadcast_text = Text("Broadcast across G genes", font_size=22)
        broadcast_text.next_to(brace, RIGHT, buff=0.2)

        self.play(GrowFromCenter(brace), Write(broadcast_text))
        self.wait(1)

        # --- Broadcasted copies ---
        broadcasted = VGroup(
            sample_vec.copy(),
            sample_vec.copy(),
            sample_vec.copy()
        ).arrange(DOWN, buff=0.25)

        # Place in frame, under sample_group
        broadcasted.next_to(sample_group, DOWN, buff=1)

        broadcast_title = Text(
            "Broadcasted Sample Embeddings", 
            font_size=24
        ).next_to(broadcasted, UP, buff=0.3)

        self.play(
            TransformFromCopy(sample_vec, broadcasted),
            Write(broadcast_title)
        )
        self.wait(1)

        # --- Summation ---
        plus1 = Text("+", font_size=36).next_to(ree_vec, RIGHT, buff=0.3)
        plus2 = Text("+", font_size=36).next_to(gene_vec, RIGHT, buff=0.3)

        self.play(Write(plus1), Write(plus2))

        summed_vec = Matrix([[1.8],[1.0],[1.5],[1.1],[2.0]])
        summed_group = VGroup(
            Text("Summed Token Embedding", font_size=24),
            summed_vec
        ).arrange(DOWN, buff=0.3)

        # Keep in-frame
        summed_group.next_to(broadcasted, DOWN, buff=1)

        self.play(
            Transform(ree_vec.copy(), summed_vec),
            Transform(gene_vec.copy(), summed_vec),
            Transform(sample_vec.copy(), summed_vec),
        )
        self.wait(1)

        self.play(FadeIn(summed_group))
        self.wait(2)
