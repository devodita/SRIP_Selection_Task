    # Train the model
    history=model.fit(
        train_generator,
        epochs=5,  # Adjust the number of epochs as needed
        validation_data=validation_generator,
    )
    
    
    # Plot the training and validation loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()


#Add this to plot the Loss Function
