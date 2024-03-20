from manim import Scene,Text,Write,ReplacementTransform,FadeOut,UP,Matrix
import numpy as np



# 
# sudo apt-get install texlive-latex-extra # 300MB


# manim -pql BroadcastinExample.py BroadcastingExample -r 1280,720
class BroadcastingExample(Scene):
    def construct(self):
        # Create tensors
        latent_codes = np.random.randn(4, 512, 10)
        speed_embeddings = np.random.randn(64)

        # Create Manim arrays
        latent_codes_array = Matrix(latent_codes.astype(int), v_buff=0.5, h_buff=1)
        speed_embeddings_array = Matrix(speed_embeddings.astype(int), v_buff=0.5)

        # Create text labels
        latent_codes_text = Text("Latent Codes").scale(0.7).next_to(latent_codes_array, UP)
        speed_embeddings_text = Text("Speed Embeddings").scale(0.7).next_to(speed_embeddings_array, UP)

        # Animate the creation of arrays and labels
        self.play(
            Write(latent_codes_array),
            Write(speed_embeddings_array),
            Write(latent_codes_text),
            Write(speed_embeddings_text),
        )
        self.wait(2)

        # Expand the speed embeddings
        expanded_speed_embeddings = np.expand_dims(np.expand_dims(speed_embeddings, axis=0), axis=-1)
        expanded_speed_embeddings = np.tile(expanded_speed_embeddings, (4, 1, 10))
        expanded_speed_embeddings_array = Matrix(expanded_speed_embeddings.astype(int), v_buff=0.5, h_buff=1)
        expanded_speed_embeddings_text = Text("Expanded Speed Embeddings").scale(0.7).next_to(expanded_speed_embeddings_array, UP)

        # Animate the expansion of speed embeddings
        self.play(
            ReplacementTransform(speed_embeddings_array, expanded_speed_embeddings_array),
            ReplacementTransform(speed_embeddings_text, expanded_speed_embeddings_text),
        )
        self.wait(2)

        # Perform broadcasting and element-wise addition
        combined_features = latent_codes + expanded_speed_embeddings
        combined_features_array = Matrix(combined_features.astype(int), v_buff=0.5, h_buff=1)
        combined_features_text = Text("Combined Features").scale(0.7).next_to(combined_features_array, UP)

        # Animate the broadcasting and addition
        self.play(
            ReplacementTransform(latent_codes_array, combined_features_array),
            ReplacementTransform(expanded_speed_embeddings_array, combined_features_array),
            ReplacementTransform(latent_codes_text, combined_features_text),
            FadeOut(expanded_speed_embeddings_text),
        )
        self.wait(2)

        # Show the final result
        self.play(combined_features_array.animate.scale(1.2))
        self.wait(3)