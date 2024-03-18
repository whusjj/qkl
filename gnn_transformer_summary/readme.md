# Please refer to the training process

```Python
def train(args, train_datasets, model, tokenizer,pool):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_samplers = [RandomSampler(train_dataset) for train_dataset in train_datasets]
    
    train_dataloaders = [cycle(DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)) for train_dataset,train_sampler in zip(train_datasets,train_samplers)]
    t_total = args.max_steps
    model.to(args.device)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()     
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last, map_location="cpu"))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last, map_location="cpu"))    
    if args.local_rank == 0:
        torch.distributed.barrier()         
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank%args.gpu_per_node],
                                                          output_device=args.local_rank%args.gpu_per_node,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", sum([len(train_dataset) for train_dataset in train_datasets])* (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    
    global_step = args.start_step
    step=0
    tr_loss, logging_loss,avg_loss,tr_nb = 0.0, 0.0,0.0,0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    probs=[len(x) for x in train_datasets]
    probs=[x/sum(probs) for x in probs]
    probs=[x**0.7 for x in probs]
    probs=[x/sum(probs) for x in probs]
    while True: 
        train_dataloader=np.random.choice(train_dataloaders, 1, p=probs)[0]
        step+=1
        batch=next(train_dataloader)
        source_ids= batch.to(args.device)
        model.train()
        loss = model(source_ids)

        if args.n_gpu > 1:
            loss = loss.mean() # mean() to average on multi-gpu parallel training

            
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        tr_loss += loss.item()


        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  
            global_step += 1
            output_flag=True
            avg_loss=round((tr_loss - logging_loss) /(global_step- tr_nb),6)

            if global_step % 100 == 0:
                logger.info("  steps: %s loss: %s", global_step, round(avg_loss,6))
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logging_loss = tr_loss
                tr_nb=global_step

            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                checkpoint_prefix = 'checkpoint'
                results = evaluate(args, model, tokenizer,pool=pool,eval_when_training=True)
                for key, value in results.items():
                    logger.info("  %s = %s", key, round(value,6))                    
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, '{}-{}-{}'.format(checkpoint_prefix, global_step,round(results['loss'],6)))

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module.encoder if hasattr(model,'module') else model.encoder  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)

                # _rotate_checkpoints(args, checkpoint_prefix)

                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save.save_pretrained(last_output_dir)
                idx_file = os.path.join(last_output_dir, 'idx_file.txt')
                with open(idx_file, 'w', encoding='utf-8') as idxf:
                    idxf.write(str(0) + '\n')

                torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", last_output_dir)

                step_file = os.path.join(last_output_dir, 'step_file.txt')
                with open(step_file, 'w', encoding='utf-8') as stepf:
                    stepf.write(str(global_step) + '\n')

            if args.max_steps > 0 and global_step > args.max_steps:
                break
    return global_step, tr_loss


def evaluate(args, model, tokenizer, prefix="",pool=None,eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_datasets = load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_samplers = [SequentialSampler(eval_dataset) for eval_dataset in eval_datasets]
    eval_dataloaders = [DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size) for eval_dataset,eval_sampler in zip(eval_datasets,eval_samplers)]

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    #logger.info("***** Running evaluation {} *****".format(prefix))
    #logger.info("  Num examples = %d", len(eval_dataset))
    #logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    for eval_dataloader in eval_dataloaders:
        for batch in eval_dataloader:
            source_ids= batch.to(args.device)
            with torch.no_grad():
                loss = model(source_ids)
                if args.n_gpu > 1:
                    loss = loss.mean()
                eval_loss += loss.item()

            nb_eval_steps += 1

    result = {
        "loss": eval_loss / nb_eval_steps,
    }

    return result

```
