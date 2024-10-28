from molclip.config import DataConfig, MolClipConfig, MolConfig, TextConfig, TrainConfig
from molclip.feature import get_mol_graph
from molclip.model import MolClip
from molclip.utils import fix_seeds


def main():
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    texts = [
        "aspirin",  # correct
        "This molecule is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce pain, fever, and inflammation",  # correct
        "This molecule appears as a clear, colorless liquid with a characteristic aromatic odor.",  # incorrect
    ]

    config = MolClipConfig(
        data=DataConfig(),
        text=TextConfig(),
        mol=MolConfig(),
        train=TrainConfig(),
    )

    fix_seeds()
    ckpt_path = "molclip/0.1.0/checkpoints/epoch=29-step=24780.ckpt"
    model = MolClip.load_from_checkpoint(ckpt_path, config=config)
    # model = MolClip(config).cuda()

    mol_graph = get_mol_graph(smiles)
    mol_embed = model.mol_encoder(mol_graph.cuda())

    text_embeds = []
    tokenizer = model.get_tokenizer()
    input_ids = tokenizer(texts, padding="max_length", max_length=config.data.max_length, return_tensors="pt")[
        "input_ids"
    ]
    text_embeds = model.text_encoder.forward(input_ids.cuda())  # type: ignore

    print("Embeddings:")
    print(f"Molecule: {mol_embed}")
    for text, text_embed in zip(texts, text_embeds):
        print(f"Text: {text}\nEmbedding: {text_embed}\n")

    print(f"Molecule: {smiles}")
    for text, text_embed in zip(texts, text_embeds):
        similarity = text_embed @ mol_embed.T
        print(f"Text: {text}\nSimilarity: {similarity.item()}\n")


if __name__ == "__main__":
    main()
