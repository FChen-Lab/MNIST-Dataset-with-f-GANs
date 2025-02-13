import torch
import os

if torch.cuda.is_available():
    device = torch.device('cuda')    
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
    map_location=torch.device('cpu')

def D_train(x, G, D, D_optimizer, criterion):
    #=======================Train the discriminator=======================#
    D.zero_grad()
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.to(device), y_real.to(device)

    D_output = torch.sigmoid(D(x_real))
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    z = torch.randn(x.shape[0], 100).to(device)
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).to(device)


    D_output =  torch.sigmoid(D(x_fake))

    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return  D_loss.data.item()


def G_train(x, G, D, G_optimizer, criterion):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).to(device)
    y = torch.ones(x.shape[0], 1).to(device)

    G_output = G(z)
    D_output = torch.sigmoid(D(G_output))
    G_loss = criterion(D_output, y)
    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    #torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
    G_optimizer.step()

    return G_loss.data.item()


def D_train_bis(x, G, D, D_optimizer, Q_criterion, V_criterion):

    D.zero_grad()

     # train discriminator on real
    x_real = x
    x_real = x_real.to(device)
    D_real_score = D(x_real)
    D_real_loss = -(V_criterion(D_real_score))  ##   >0
    #D_real_loss.backward(retain_graph=True)

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100).to(device)
    x_fake = G(z)
    D_fake_score = D(x_fake)
    D_fake_loss = -(Q_criterion(D_fake_score))  ##  >0
    #D_fake_loss.backward()

    # gradient backprop & optimize ONLY D's parameters
    D_loss = (D_real_loss + D_fake_loss)   ##>0
    D_loss.backward()
    #torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
    D_optimizer.step()

    return  D_loss.data.item()

def G_train_bis(x, G, D, G_optimizer, Q_criterion, V_criterion):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).to(device)

    G_output = G(z)
    D_fake_score = D(G_output)
    G_loss = -V_criterion(D_fake_score)  # >0
    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    #torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
    G_optimizer.step()

    return G_loss.data.item()


def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, D, folder):
    # Charger les poids du générateur
    ckpt_G = torch.load(os.path.join(folder,'G.pth'), map_location=torch.device('cpu'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt_G.items()})
    
    # Charger les poids du discriminateur
    ckpt_D = torch.load(os.path.join(folder,'D.pth'), map_location=torch.device('cpu'))
    D.load_state_dict({k.replace('module.', ''): v for k, v in ckpt_D.items()})
    
    return G, D
