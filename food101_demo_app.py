import os
import argparse
import torch
from torchvision import transforms
import json
import gradio as gr
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
import io
import base64
from improved_training import get_model

def parse_args():
    parser = argparse.ArgumentParser(description='Food-101 Classifier Demo App')
    parser.add_argument('--model-path', type=str, default='./runs/food101/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--class-names', type=str, default='./runs/food101/class_names.json', help='Path to class names JSON file')
    parser.add_argument('--model-type', type=str, default='efficientnet_b0', 
                        choices=['resnet18', 'resnet34', 'resnet50', 'densenet121', 'efficientnet_b0', 'efficientnet_b2'], help='Model architecture')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--share', action='store_true', help='Create a public link for the app')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the app on')
    parser.add_argument('--theme', type=str, default='soft', choices=['soft', 'default', 'grass', 'peach'], help='Gradio theme')
    
    return parser.parse_args()

def load_model(model_path, model_type, class_names_path, device):
    """Load the trained model"""
    # Load class names
    try:
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
    except Exception as e:
        print(f"Error loading class names: {e}")
        # Fallback to creating a list of 101 food categories
        class_names = [f"Class_{i}" for i in range(101)]
    
    # Load model
    try:
        model = get_model(model_type, num_classes=len(class_names), dropout_rate=0.0)  # No dropout during inference
    except Exception as e:
        print(f"Error creating model: {e}. Trying original implementation...")
        # Fallback to original get_model implementation if available
        from food101_classifier import get_model as original_get_model
        model = original_get_model(model_type, num_classes=len(class_names))
    
    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Using model with random weights.")
    
    model = model.to(device)
    model.eval()
    
    return model, class_names

def predict_image(image, model, class_names, device, top_k=5):
    """Make a prediction for an image and create visualizations"""
    start_time = time.time()
    
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    
    # Keep a copy of the original image for display
    original_image = image.copy()
    
    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # Get top k predictions
    top_probs, top_indices = torch.topk(probabilities[0], top_k)
    
    # Convert to lists
    top_probs = top_probs.cpu().numpy().tolist()
    top_indices = top_indices.cpu().numpy().tolist()
    
    # Get class names
    top_labels = [class_names[idx] for idx in top_indices]
    
    # Make them pretty (replace underscores with spaces and capitalize)
    pretty_labels = [label.replace('_', ' ').title() for label in top_labels]
    
    # Format as dictionary for Gradio
    results = {label: float(prob) for label, prob in zip(pretty_labels, top_probs)}
    
    # Calculate inference time
    inference_time = time.time() - start_time
    
    # Create a more detailed visualization
    fig = plt.figure(figsize=(10, 6))
    
    # Plot the image on the left
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(original_image)
    ax1.set_title("Input Image")
    ax1.axis("off")
    
    # Plot the predictions on the right
    ax2 = fig.add_subplot(1, 2, 2)
    y_pos = range(len(pretty_labels))
    bars = ax2.barh(y_pos, [prob * 100 for prob in top_probs], align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(pretty_labels)
    ax2.set_xlabel("Probability (%)")
    ax2.set_title(f"Top {top_k} Predictions")
    
    # Add percentage values on the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{top_probs[i]*100:.1f}%', ha='left', va='center')
    
    # Add inference time as text
    fig.text(0.5, 0.01, f"Inference time: {inference_time:.3f} seconds", ha='center')
    
    # Save the figure to a buffer
    buf = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    
    # Convert buffer to base64 image
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    # HTML for displaying the image
    html_content = f'<img src="data:image/png;base64,{img_str}" alt="Prediction visualization">'
    
    return results, html_content, pretty_labels[0]  # Return top prediction as text

def main():
    args = parse_args()
    
    # Display CUDA information if available
    device = torch.device(args.device)
    if device.type == 'cuda':
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU for inference")
    
    # Load model
    model, class_names = load_model(args.model_path, args.model_type, args.class_names, device)
    print(f"Number of classes: {len(class_names)}")
    
    # Get theme
    try:
        theme = getattr(gr.themes, args.theme.capitalize())()
    except:
        theme = gr.themes.Soft()
    
    # Create example directory if it doesn't exist
    example_dir = "example_images"
    os.makedirs(example_dir, exist_ok=True)
    
    # Define Gradio interface with enhanced features
    with gr.Blocks(theme=theme, title="Food-101 Classifier - Interactive Demo") as demo:
        gr.Markdown("""
        # 🍔 Food-101 Classifier
        
        Upload or take a photo of food to classify it among 101 food categories.
        
        This model was trained on the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) using PyTorch.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input components
                input_image = gr.Image(type="pil", label="Upload or take a food photo")
                camera_button = gr.Button("Take Photo 📷", variant="primary")
                submit_button = gr.Button("Classify Food 🔍", variant="secondary")
                clear_button = gr.Button("Clear ❌")
                
                # Model info
                with gr.Accordion("Model Information", open=False):
                    gr.Markdown(f"""
                    - **Model Architecture**: {args.model_type}
                    - **Device**: {args.device}
                    - **Categories**: 101 food classes
                    """)
                
            with gr.Column(scale=1):
                # Output components
                label = gr.Label(label="Food Classification", num_top_classes=5)
                result_text = gr.Textbox(label="Top Prediction")
                visualization = gr.HTML(label="Visualization")
                
        # Event handlers
        submit_button.click(
            fn=lambda img: predict_image(img, model, class_names, device),
            inputs=input_image,
            outputs=[label, visualization, result_text]
        )
        
        input_image.change(
            fn=lambda img: predict_image(img, model, class_names, device) if img is not None else (None, None, None),
            inputs=input_image,
            outputs=[label, visualization, result_text]
        )
        
        clear_button.click(
            fn=lambda: (None, None, None),
            inputs=[],
            outputs=[input_image, label, visualization]
        )
        
        # Examples
        example_paths = []
        example_food_classes = ["pizza", "ice_cream", "hamburger", "sushi", "pasta", "steak"]
        
        for food in example_food_classes:
            if os.path.exists(os.path.join(example_dir, f"{food}.jpg")):
                example_paths.append(os.path.join(example_dir, f"{food}.jpg"))
        
        if example_paths:
            gr.Examples(
                examples=example_paths,
                inputs=input_image,
                outputs=[label, visualization, result_text],
                fn=lambda img: predict_image(img, model, class_names, device),
                cache_examples=True
            )
        
        gr.Markdown("""
        ## How It Works
        
        1. Upload an image or take a photo of food
        2. The model will classify the food into one of 101 categories
        3. Results show the top 5 predictions with confidence scores
        
        The model was trained using transfer learning on a pre-trained network.
        """)
    
    # Launch the app
    demo.launch(share=args.share, server_port=args.port)

if __name__ == '__main__':
    main()
