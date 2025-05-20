from simple_demo import Model, DataLoader


def test_model():
    model = Model(128)
    dataLoader = DataLoader(batch_size=1, max_length=200, vocab_size=20)
    for i, batch in enumerate(dataLoader):
        print(batch.shape)
        probs = model(batch)
        print(probs)
        print(probs.shape)
        break

if __name__ == "__main__":
    test_model()