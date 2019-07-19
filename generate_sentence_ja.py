from bert_juman import BertWithJumanModel


def main():
    bert = BertWithJumanModel('./Japanese_Model')
    emb = bert.get_sentence_embedding("こんにちは")
    print(emb)


if __name__ == "__main__":
    main()