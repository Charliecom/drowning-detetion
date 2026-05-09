import hydra


@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg):
    print(cfg)
    print(type(cfg))
    print(cfg.model.model)


if __name__ == "__main__":
    main()
