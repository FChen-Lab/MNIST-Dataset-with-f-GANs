import torch 
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt  # Import pour tracer les courbes
import torchvision

from model import Generator, Discriminator, VLOSS, QLOSS
from UtilsBIS import D_train, G_train, save_models, D_train_bis, G_train_bis, load_model

# Définition du device
if torch.cuda.is_available():
    device = torch.device('cuda')    
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
    map_location = torch.device('cpu')

def save_single_image(epoch, G, device, save_dir='generated_images'):
    """ Sauvegarde une seule image générée par le générateur à chaque époque """
    os.makedirs(save_dir, exist_ok=True)
    
    # Générer une seule image avec le générateur
    z = torch.randn(1, 100).to(device)  # Vecteur aléatoire pour une seule image
    generated_image = G(z).detach().cpu()  # Extraire l'image générée
    generated_image = generated_image.view(28, 28)  # Reshape pour une image MNIST
    
    # Affichage et sauvegarde
    plt.imshow(generated_image, cmap='gray')  # MNIST est en niveaux de gris
    plt.axis('off')  # Supprime les axes
    plt.savefig(f"{save_dir}/generated_epoch_{epoch}.png")
    plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAN.')
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD")
    parser.add_argument("--divergence", type=str, default="JSD", 
                        help="f-divergence type (GAN, KLD, JSD, etc.).")

    args = parser.parse_args()

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Chargement des données
    print('Dataset loading...')
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')

    # Chargement des modèles
    print('Model Loading...')
    mnist_dim = 784
    G = Generator(g_output_dim=mnist_dim).to(device)
    D = Discriminator(mnist_dim).to(device)
    print('Model loaded.')

    # Définition de la fonction de perte et des optimiseurs
    criterion = nn.BCELoss()
    Q_criterion = QLOSS(args.divergence).to(device)
    V_criterion = VLOSS(args.divergence).to(device)
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr)

    # Listes pour stocker les pertes
    D_losses = []
    G_losses = []

    print('Start Training :')

    n_epoch = args.epochs
    n_epoch_pretrain = n_epoch - 25
    #1#---------------------------------Prétrain AVEC GAN 
    for epoch in trange(1, n_epoch_pretrain + 1, leave=True):           
        epoch_D_loss = 0
        epoch_G_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim).to(device)
            
            # Entraînement du Discriminateur et récupération de la perte
            D_loss = D_train(x, G, D, D_optimizer, criterion)
            epoch_D_loss += D_loss
            
            # Entraînement du Générateur et récupération de la perte
            G_loss = G_train(x, G, D, G_optimizer, criterion)
            epoch_G_loss += G_loss

        save_single_image(epoch, G, device)
        print("D_loss =", D_loss)
        print("G_loss =", G_loss)
        # Moyenne des pertes pour l'epoch
        D_losses.append(epoch_D_loss / len(train_loader))
        G_losses.append(epoch_G_loss / len(train_loader))

        # Sauvegarde des modèles périodiquement
        if epoch % 10 == 0:
            save_models(G, D, 'checkpoints')

    # Charger les poids sauvegardés du générateur et du discriminateur
    G, D = load_model(G, D, 'checkpoints') 
    G = G.to(device)
    D = D.to(device)
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr/10)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr/10)


    #2#---------------------------FineTuning AVEC Discriminateur
    for epoch in trange(n_epoch_pretrain + 1, n_epoch_pretrain + 6, leave=True):           
        epoch_D_loss = 0
        epoch_G_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim).to(device)
            
            # Entraînement du Discriminateur et récupération de la perte
            D_loss = D_train_bis(x, G, D, D_optimizer, Q_criterion, V_criterion)
            epoch_D_loss += D_loss
            
            # Entraînement du Générateur et récupération de la perte
            epoch_G_loss += G_loss

        save_single_image(epoch, G, device)
        print("D_loss =", D_loss)
        print("G_loss =", G_loss)
        # Moyenne des pertes pour l'epoch
        D_losses.append(epoch_D_loss / len(train_loader))
        G_losses.append(epoch_G_loss / len(train_loader))

        # Sauvegarde des modèles périodiquement
        if epoch % 5 == 0:
            save_models(G, D, 'checkpoints')

    G, D = load_model(G, D, 'checkpoints') 
    G = G.to(device)
    D = D.to(device)
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr/10)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr/10)




    #3#----------------------FineTuning AVEC Générateur et Discriminateur
    for epoch in trange(n_epoch_pretrain + 6, n_epoch_pretrain + 26, leave=True):           
        epoch_D_loss = 0
        epoch_G_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim).to(device)
            
            # Entraînement du Discriminateur et récupération de la perte
            D_loss = D_train_bis(x, G, D, D_optimizer, Q_criterion, V_criterion)
            epoch_D_loss += D_loss
            
            # Entraînement du Générateur et récupération de la perte
            G_loss = G_train_bis(x, G, D, G_optimizer, Q_criterion, V_criterion)
            epoch_G_loss += G_loss

        save_single_image(epoch, G, device)
        print("D_loss =", D_loss)
        print("G_loss =", G_loss)
        # Moyenne des pertes pour l'epoch
        D_losses.append(epoch_D_loss / len(train_loader))
        G_losses.append(epoch_G_loss / len(train_loader))

        # Sauvegarde des modèles périodiquement
        if epoch % 10 == 0:
            save_models(G, D, 'checkpoints')
        

    print('Training done')

    # Création des courbes de perte
    plt.figure(figsize=(10, 5))
    plt.plot(D_losses, label='Discriminator Loss')
    plt.plot(G_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss During Training')
    plt.legend()

    # Sauvegarde du graphe au lieu de l'afficher
    plt.savefig("training_loss_graph.png", dpi=300)  # Fichier PNG de haute qualité
    plt.close()  # Ferme le graphe pour libérer de la mémoire