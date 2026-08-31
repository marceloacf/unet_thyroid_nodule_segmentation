import tensorflow as tf


def configurar_gpu():
    gpus = tf.config.list_physical_devices("GPU")

    print("GPUs encontradas:", gpus)

    if not gpus:
        print("Nenhuma GPU encontrada. O treinamento será executado em CPU.")
        return

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        logical_gpus = tf.config.list_logical_devices("GPU")

        print(f"{len(gpus)} GPU física(s), {len(logical_gpus)} GPU lógica(s).")
        print("Memory growth habilitado com sucesso.")

    except RuntimeError as e:
        print("Erro ao configurar memory growth da GPU:")
        print(e)

# import tensorflow as tf


# def configurar_gpu():
#     gpus = tf.config.list_physical_devices("GPU")

#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)