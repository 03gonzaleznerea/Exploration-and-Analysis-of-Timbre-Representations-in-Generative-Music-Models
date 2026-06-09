import torch
from torch import nn


class TSModel(torch.nn.Module):
    """Wrapper para trabajar con el checkpoint TorchScript de AFTER.

    Esta clase mantiene la misma interfaz utilizada en los experimentos previos:
    - ae_encode: audio -> representación latente
    - timbre: representación latente -> embedding tímbrico
    """

    def __init__(self, ts_model, emb_model=None):
        super().__init__()
        self.model = ts_model
        self.emb_model = emb_model

    def ae_encode(self, x):
        """Codifica el audio en el espacio latente del autoencoder."""
        if len(x.shape) > 1:
            x = x.reshape(x.shape[0], 1, -1)
        else:
            x = x.reshape(1, 1, -1)

        if self.emb_model is not None:
            return self.emb_model.encode(x)

        return self.model.emb_model_structure.encode(x)

    def ae_decode(self, z):
        """Reconstruye audio a partir de una representación latente."""
        if self.emb_model is not None:
            return self.emb_model.decode(z).squeeze().cpu()

        return self.model.emb_model_structure.decode(z).squeeze().cpu()

    def timbre(self, z):
        """Extrae el embedding tímbrico a partir de la representación latente."""
        return self.model.encoder.forward_stream(z)

    def structure(self, z):
        """Extrae el embedding estructural a partir de la representación latente."""
        return self.model.encoder_time.forward_stream(z)

    def sample(
        self,
        noise,
        z_structure,
        z_timbre,
        guidance_timbre,
        guidance_structure,
        nb_steps,
    ):
        """Genera audio a partir de ruido y condicionamientos."""
        self.model.set_guidance_timbre(guidance_timbre)
        self.model.set_guidance_structure(guidance_structure)
        self.model.set_nb_steps(nb_steps)

        zout = self.model.sample(
            noise,
            time_cond=z_structure,
            cond=z_timbre,
        )

        return zout


class AFTER_Timbre(nn.Module):
    """Extractor de embeddings tímbricos de AFTER para Synesis.

    Esta versión carga directamente el checkpoint TorchScript (.ts) utilizado
    en los experimentos previos de visualización, en lugar de reconstruir el
    modelo desde archivos .gin y checkpoints separados.
    """

    def __init__(self, feature_extractor=True, extract_kws={}):
        super(AFTER_Timbre, self).__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Ruta fija al checkpoint TorchScript de AFTER.
        ts_path = "/content/drive/MyDrive/TFG_AFTER/checkpoints/after.audio.instruments.ts"

        if not torch.cuda.is_available():
            print("Aviso: AFTER_Timbre se ejecutará en CPU. La extracción puede ser lenta.")

        # En los experimentos previos se usó el autoencoder integrado dentro del .ts.
        autoencoder_path = None

        # Carga del modelo TorchScript.
        ts_model = torch.jit.load(
            ts_path,
            map_location=self.device,
        )
        ts_model = ts_model.eval().to(self.device)

        # Carga opcional de un autoencoder externo.
        if autoencoder_path is not None:
            emb_model = torch.jit.load(
                autoencoder_path,
                map_location=self.device,
            )
            emb_model = emb_model.eval().to(self.device)
        else:
            emb_model = None

        # Misma lógica que en el notebook previo:
        # model = TSModel(ts_model, emb_model=emb_model)
        self.model = TSModel(
            ts_model,
            emb_model=emb_model,
        )
        self.model = self.model.eval().to(self.device)

    @torch.no_grad()
    def forward(self, x):
        """Extrae el embedding tímbrico a partir del audio de entrada."""
        x = x.to(self.device)

        # Audio -> representación latente.
        z = self.model.ae_encode(x)

        # Representación latente -> embedding tímbrico.
        timbre_emb = self.model.timbre(z)

        return timbre_emb.cpu()
