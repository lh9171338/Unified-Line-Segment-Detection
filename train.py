import os
import numpy as np
import copy
from tqdm import tqdm
import random
import shutil
from sklearn.metrics import confusion_matrix
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from network.lcnn import LCNN, lpn_loss_func, loi_loss_func
from network.dataset import Dataset
from config.cfg import parse
from metric.eval_mAPJ import eval_mAPJ
from metric.eval_sAP import eval_sAP


def train(model, loader, cfg, device):
    # Option
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size)

    # Summary
    writer = SummaryWriter(cfg.log_path)

    # Train
    step = 1
    best_sAP = [0.0, 0.0, 0.0]
    best_state_dict = None
    for epoch in range(1, cfg.num_epochs + 1):
        # Train
        model.train()

        for images, map_labels, metas in tqdm(loader['train'], desc='train: '):
            images = images.to(device)
            map_labels = {name: map_labels[name].to(device) for name in map_labels.keys()}
            metas = {name: [meta.to(device) for meta in metas[name]] for name in metas.keys()}

            map_preds, loi_scores, loi_labels = model(images, metas)
            lmap_loss, jmap_loss, joff_loss, cmap_loss, coff_loss, lvec_loss  = lpn_loss_func(map_preds, map_labels)
            loi_loss = loi_loss_func(loi_scores, loi_labels)

            losses = [lmap_loss, jmap_loss, joff_loss, cmap_loss, coff_loss, lvec_loss, loi_loss]
            loss = sum([weight * loss for weight, loss in zip(cfg.weights, losses)])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Visualize
            if step % cfg.print_freq == 0:
                image = images[0]
                lmap_pred = map_preds['lmap'][0]
                jmap_pred = map_preds['jmap'][0]
                cmap_pred = map_preds['cmap'][0]
                lmap_label = map_labels['lmap'][0]
                jmap_label = map_labels['jmap'][0]
                cmap_label = map_labels['cmap'][0]
                image = F.interpolate(image[None, :, :, :], (lmap_label.shape[-2], lmap_label.shape[-1]))[0]
                lr = scheduler.get_last_lr()[0]

                loi_label = loi_labels[0].detach().cpu().numpy() > 0.5
                loi_score = loi_scores[0].detach().cpu().numpy() > 0.5
                tn, fp, fn, tp = confusion_matrix(loi_label, loi_score).ravel()

                print('epoch: %d/%d | loss: %6f | lmap loss: %6f | jmap loss: %6f | joff loss: %6f | cmap loss: %6f | '
                      'coff loss: %6f | lvec loss: %6f | loi loss: %6f | lr: %e' % (epoch, cfg.num_epochs, loss.item(),
                      lmap_loss.item(), jmap_loss.item(), joff_loss.item(), cmap_loss.item(), coff_loss.item(),
                      lvec_loss.item(), loi_loss.item(), lr))
                image_list = [loader['train'].dataset.DeNormalize(image),
                              lmap_label.repeat((3, 1, 1)), lmap_pred.repeat((3, 1, 1)),
                              jmap_label.repeat((3, 1, 1)), jmap_pred.repeat((3, 1, 1)),
                              cmap_label.repeat((3, 1, 1)), cmap_pred.repeat((3, 1, 1))]
                writer.add_images('image', image_list, step, dataformats='CHW')
                writer.add_scalar('loss', loss, step)
                writer.add_scalar('lmap loss', lmap_loss, step)
                writer.add_scalar('jmap loss', jmap_loss, step)
                writer.add_scalar('joff loss', joff_loss, step)
                writer.add_scalar('cmap loss', cmap_loss, step)
                writer.add_scalar('coff loss', coff_loss, step)
                writer.add_scalar('lvec loss', lvec_loss, step)
                writer.add_scalar('loi loss', loi_loss, step)
                writer.add_scalars('metrics', {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}, step)
                writer.add_scalar('lr', lr, step)
            step += 1
        scheduler.step()

        if epoch % cfg.save_freq == 0:
            # Save model
            if cfg.save_temp:
                model_filename = os.path.join(cfg.model_path, 'epoch-%03d.pkl' % epoch)
                torch.save(model.state_dict(), model_filename)

            save_path = os.path.join(cfg.model_path, 'epoch-%03d' % epoch)
            if not os.path.isdir(save_path):
                os.mkdir(save_path)

            # Val
            model.eval()

            index = 0
            for images in tqdm(loader['val'], desc='val: '):
                images = images.to(device)
                jmaps, joffs, line_preds, line_scores = model(images)

                jmaps = jmaps.detach().cpu()
                joffs = joffs.detach().cpu()
                line_preds = [line_pred.detach().cpu() for line_pred in line_preds]
                line_scores = [line_score.detach().cpu() for line_score in line_scores]
                for i in range(len(images)):
                    jmap = jmaps[i].numpy()
                    joff = joffs[i].numpy()
                    line_pred = line_preds[i].numpy()
                    line_score = line_scores[i].numpy()

                    src_filename = loader['val'].dataset.file_list[index].split()[0]
                    filename = os.path.split(src_filename)[1]
                    image_filename = os.path.join(save_path, filename)
                    npz_filename = image_filename.replace('.png', '.npz')
                    np.savez_compressed(npz_filename, jmap=jmap, joff=joff,
                                        line_pred=line_pred, line_score=line_score)
                    index += 1

            mAPJ, P, R = eval_mAPJ(cfg.groundtruth_path, save_path)
            print(f'APJ: {mAPJ:.1f} | {P:.1f} | {R:.1f}')
            msAP, P, R, _ = eval_sAP(cfg.groundtruth_path, save_path)
            print(f'sAP: {msAP:.1f} | {P:.1f} | {R:.1f}')
            writer.add_scalar('sAP', msAP, step)

            if not cfg.save_temp:
                shutil.rmtree(save_path)

            if msAP > best_sAP[0]:
                best_sAP = [msAP, P, R]
                best_state_dict = copy.deepcopy(model.state_dict())
            print(f'best sAP: {best_sAP[0]:.1f} | {best_sAP[1]:.1f} | {best_sAP[2]:.1f}')

    writer.close()

    # Save best model
    model_filename = os.path.join(cfg.model_path, 'best.pkl')
    torch.save(best_state_dict, model_filename)


if __name__ == '__main__':
    # Parameter
    cfg = parse()
    print(cfg)
    os.makedirs(cfg.model_path, exist_ok=True)

    # Use GPU or CPU
    use_gpu = cfg.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f'cuda:{cfg.gpu}' if use_gpu else 'cpu')
    print('use_gpu: ', use_gpu)

    # Seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    if use_gpu:
        torch.cuda.manual_seed(cfg.seed)

    # Load model
    model = LCNN(cfg).to(device)

    # Load dataset
    train_dataset = Dataset(os.path.join(cfg.train_dataset_path, 'train'), cfg, with_label=True)
    val_dataset = Dataset(os.path.join(cfg.train_dataset_path, 'test'), cfg, with_label=False)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=cfg.train_batch_size,
                                 num_workers=cfg.num_workers, shuffle=True, collate_fn=train_dataset.collate)
    val_loader = Data.DataLoader(dataset=val_dataset, batch_size=cfg.test_batch_size,
                                   num_workers=cfg.num_workers, shuffle=False)
    loader = {'train': train_loader, 'val': val_loader}

    # Train network
    train(model, loader, cfg, device)
