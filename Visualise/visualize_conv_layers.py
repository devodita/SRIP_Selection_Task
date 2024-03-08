import matplotlib.pyplot as plt

#Add this to the corresponding .py to get the activation layers

# Plot the output of all convolutional layers
    sample_images, _ = next(train_generator)  # Get a batch of sample images
    layer_outputs = [layer.output for layer in model.layers[:6] if 'conv2d' in layer.name]  # Choose the first 6 convolutional layers
    activation_model = tf.keras.models.Model(inputs=model.inputs, outputs=layer_outputs)
    activations = activation_model.predict(sample_images)
    # Plot the output of each convolutional layer
    for i, activation in enumerate(activations):
        plt.figure(figsize=(8, 8))
        plt.title(f'Activation of Conv2D Layer {i + 1}')
        plt.imshow(activation[0, :, :, 0], cmap='viridis')  # Display the first channel of the activation
    plt.show()
