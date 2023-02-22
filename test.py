model.load_state_dict(torch.load('/home/viktoriia.trokhova/model_weights/model_best.pt'), strict=False)

test_dataset = myDataset_test(transform = None)
test_dataloader = torch.utils.data.DataLoader(myDataset_test(transform = None),
                                    batch_size=32,
                                    shuffle=False,
                                    num_workers=0)

model.eval()
running_loss = 0.0
running_corrects = 0.0
for inputs, labels, masks in test_dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    masks = masks.to(device)
  
    outputs, targets_, xe_loss_, gcam_losses_, imgs_feats = model(inputs, labels, masks, batch_size = inputs.size(0), dropout=nn.Dropout(0.79))

    loss = xe_loss_.mean() + 0.575 * gcam_losses_
    
    _, preds = torch.max(outputs, 1)  

    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)

epoch_loss = running_loss / 1871
epoch_acc = running_corrects.double() / 1871
print('Test loss: {:.4f}, acc: {:.4f}'.format(epoch_loss,
                                            epoch_acc))
