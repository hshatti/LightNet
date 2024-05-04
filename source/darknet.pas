unit darknet;

{$apptype console}
{$ifdef fpc}
  {$mode delphi}
  {$ModeSwitch cvar}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
  {$PackRecords C}
{$endif}
{$pointermath on}

interface
uses  SysUtils
  , TypInfo
;
const
  SECRET_NUM = -1234;

{$j+}
const
    gpu_index : longint=-1;
    benchmark : boolean = true;
{$j-}

type

  {$ifndef FPC}
     PPtrInt = ^PtrInt;
     PtrInt = IntPtr;
     TStringArray = TArray<string>;
     PPByte = ^PByte;
     {$ifdef ANDROID}
     longint = integer;
     {$endif}
  {$endif}
  PPSingle  = ^PSingle;
  {.$if not declared(system.psize_t)}
    Psize_t = ^size_t;
  {.$endif}

  {.$if not declared(system.size_t)}
    size_t = NativeUInt;
  {.$endif}

  {$if not declared(pthread_t)}
    pthread_t = ^thread_t;
  {$endif}

  {$if not declared(thread_t)}
    thread_t = TThreadId;
  {$endif}

  {$if not declared(PFile)}
    PFile = ^File;
  {$endif}

  {$if not declared(clock_t)}
    clock_t = int64;
  {$endif}
  {$ifdef NO_POINTERS}
  TSingles     = TArray<single>;
  TSingles2D   = TArray<TSingles>;
  TIntegers    = TArray<longint>;
  TIntegers2D  = TArray<TIntegers>;
  {$else}
  PSingles     = ^TSingles;
  PIntegers    = ^TIntegers;
  TSingles2d   = PSingles;
  TSingles     = PSingle;
  TIntegers2d  = PIntegers;
  TIntegers    = PLongint;
  {$endif}

  PUNUSED_ENUM_TYPE = ^TUNUSED_ENUM_TYPE;
  TUNUSED_ENUM_TYPE = (UNUSED_DEF_VAL);

  { TSinglesHelper }

  TSinglesHelper = record helper for TSingles
    constructor Create(const aCount:size_t);
    procedure ReAllocate(const aCount:size_t);
    function Count():size_t;
    function High():PtrInt;
    procedure Free;
    function toString(const sep:string=', ';N :longint=-1):string;
  end;

  { TSingles2dHelper }

  TSingles2dHelper = record helper for TSingles2d
    class function Create(const aCount:size_t):TSingles2d;static;
    procedure ReAllocate(const aCount:size_t);
    function Count():size_t;
    function High():PtrInt;
    procedure Free;
    function toString(const sep: string=#13#10; N: longint=-1): string;
  end;

  { TIntegersHelper }

  TIntegersHelper = record helper for TIntegers
    constructor Create(const aCount:size_t);
    procedure ReAllocate(const aCount:size_t);
    function Count():size_t;
    function High():PtrInt;
    procedure Free;
    function toString(const sep:string=', '):string;
  end;

  { TIntegers2dHelper }

  TIntegers2dHelper = record helper for TIntegers2d
    class function Create(const aCount:size_t):TIntegers2d;static;
    procedure ReAllocate(const aCount:size_t);
    function Count():size_t;
    function High():PtrInt;
    procedure Free;
    function toString(const sep:string=', '):string;
  end;

  PMetadata = ^TMetadata;
  TMetadata = record
      classes : longint;
      names : TArray<string>;
    end;

  PTree = ^TTree;

  TTree = record
      leaf : TArray<longint>;//TIntegers;
      n : longint;
      parent : TArray<longint>;// TIntegers;
      child :  TArray<longint>;//TIntegers;
      group :  TArray<longint>;//TIntegers;
      name : TArray<string>;
      groups : longint;//longint;
      group_size :  TArray<longint>;//TIntegers;
      group_offset :  TArray<longint>;//TIntegers;
  end;


  PActivation = ^TActivation;
  TActivation = (
    acLOGISTIC, acRELU, acRELU6, acRELIE, acLINEAR, acRAMP, acTANH, acPLSE,
    acREVLEAKY, acLEAKY, acELU, acLOGGY, acSTAIR, acHARDTAN, acLHTAN, acSELU,
    acGELU, acSWISH, acMISH, acHARD_MISH, acNORM_CHAN, acNORM_CHAN_SOFTMAX,
    acNORM_CHAN_SOFTMAX_MAXVAL
  );

  TIOULoss = (
      ilIOU, ilGIOU, ilMSE, ilDIOU, ilCIOU
  ) ;

  // parser.h
  TNMSKind = (
      nmsDEFAULT_NMS, nmsGREEDY_NMS, nmsDIOU_NMS, nmsCORNERS_NMS
  ) ;

  // parser.h
  TYOLOPoint = (
      ypYOLO_CENTER = 1 shl 0, ypYOLO_LEFT_TOP = 1 shl 1, ypYOLO_RIGHT_BOTTOM = 1 shl 2
  ) ;

  // parser.h
  TWeightsType = (
      wtNO_WEIGHTS, wtPER_FEATURE, wtPER_CHANNEL
  );

  // parser.h
  TWeightsNormalization = (
      wnNO_NORMALIZATION, wnRELU_NORMALIZATION, wnSOFTMAX_NORMALIZATION
  );

  PImType = ^TImType;
  TImType = (imtPNG, imtBMP, imtTGA, imtJPG);

  PBinary_Activation = ^TBinary_Activation;
  TBinary_Activation = (bacMULT, bacADD, bacSUB, bacDIV);

  PContrastiveParams = ^TContrastiveParams;
  TContrastiveParams = record
    sim, exp_sim, P:single;
    i, j:longint;
    time_step_i, time_step_j: longint;

  end;

  PLayerType = ^TLayerType;
  TLayerType = (
    ltCONVOLUTIONAL,
    ltDECONVOLUTIONAL,
    ltCONNECTED,
    ltMAXPOOL,
    ltLOCAL_AVGPOOL,
    ltSOFTMAX,
    ltDETECTION,
    ltDROPOUT,
    ltCROP,
    ltROUTE,
    ltCOST,
    ltNORMALIZATION,
    ltAVGPOOL,
    ltLOCAL,
    ltSHORTCUT,
    ltScaleChannels,
    ltSAM,
    ltACTIVE,
    ltRNN,
    ltGRU,
    ltLSTM,
    ltConvLSTM,
    ltHISTORY,
    ltCRNN,
    ltBATCHNORM,
    ltNETWORK,
    ltXNOR,
    ltREGION,
    ltYOLO,
    ltGaussianYOLO,
    ltISEG,
    ltREORG,
    ltREORG_OLD,
    ltUPSAMPLE,
    ltLOGXENT,
    ltL2NORM,
    ltEMPTY,
    ltBLANK,
    ltCONTRASTIVE,
    ltIMPLICIT
    );

  PCostType = ^TCostType;
  TCostType = (ctSSE, ctMASKED, ctL1, ctSEG, ctSMOOTH, ctWGAN);

  PDataType = ^TDatatype;
  TDataType = (
    dtCLASSIFICATION_DATA,      dtDETECTION_DATA, dtCAPTCHA_DATA,
    dtREGION_DATA,     dtIMAGE_DATA,        dtCOMPARE_DATA,   dtWRITING_DATA,
    dtSWAG_DATA,       dtTAG_DATA,          dtOLD_CLASSIFICATION_DATA,
    dtSTUDY_DATA,      dtDET_DATA,          dtSUPER_DATA,     dtLETTERBOX_DATA,
    dtREGRESSION_DATA, dtSEGMENTATION_DATA, dtINSTANCE_DATA,
    dtISEG_DATA);

  PUpdateArgs = ^TUpdateArgs;
  TUpdateArgs = record
      batch : longint;
      learning_rate : Single;
      momentum : Single;
      decay : Single;
      adam : boolean;
      B1 : Single;
      B2 : Single;
      eps : Single;
      t : longint;
    end;

  PAugmentArgs = ^TAugmentArgs;
  TAugmentArgs = record
      w : longint;
      h : longint;
      scale : Single;
      rad : Single;
      dx : Single;
      dy : Single;
      aspect : Single;
    end;

  PPImageData = ^PImageData;
  PImageData = ^TImageData;
  TImageData = record
      w : longint;
      h : longint;
      c : longint;
      data : TArray<Single>;
    end;


  PPBox= ^PBox;
  PBox = ^TBox;
  TBox = record
      x : Single;
      y : Single;
      w : Single;
      h : Single;
    end;

  PBoxAbs = ^TBoxabs;
  TBoxAbs = record
      left : single;
      right : single;
      top : single;
      bot : single;
    end;

  PDxrep = ^TDxrep;
  TDxrep = record
      dt : single;
      db : single;
      dl : single;
      dr : single;
    end;

  PIOUs = ^TIOUs;
  TIOUs = record
      iou : single;
      giou : single;
      diou : single;
      ciou : single;
      dx_iou : TDxrep;
      dx_giou : TDxrep;
    end;


  PDetection = ^TDetection;
  TDetection = record
      bbox : TBox;
      classes, best_class_idx: longint;

      prob, mask : TArray<Single>;//TSingles;
      //mask : //TSingles;
      objectness : Single;
      sort_class : longint;
      uc :TArray<single>;
      points : longint;
      embeddings : TArray<single>;
      embedding_size : longint;
      sim : single;
      track_id : longint;
    end;

  PDetNumPair  = ^TDetNumPair;
  TDetNumPair = record
    num :longint;
    dets :TArray<TDetection>;
  end;


  PMatrix = ^TMatrix;
  TMatrix = record
      rows : longint;
      cols : longint;
      vals : TArray<TArray<Single>>//TSingles2d;
      //vals : TArray<TSingles>;
    end;

  PData = ^TData;
  TData = record
      w : longint;
      h : longint;
      X : TMatrix;
      y : TMatrix;
      shallow : boolean;
      num_boxes : TIntegers;
      boxes : PPbox;
    end;

  PLoadArgs = ^TLoadArgs;
  TLoadArgs = record
       threads:           longint;
       paths:             TArray<string>;
       path:              string;
       n:                 longint;
       m:                 longint;
       labels:            TArray<string>;
       h:                 longint;
       w:                 longint;
       c:                 longint;
       out_w:             longint;
       out_h:             longint;
       nh:                longint;
       nw:                longint;
       num_boxes:         longint;
       truth_size:        longint;
       min, max, size:    longint;
       classes:           longint;
       background:        longint;
       scale:             longint;
       center:            boolean;
       coords:            longint;
       mini_batch:        longint;
       track:             longint;
       augment_speed:     longint;
       letter_box:        longint;
       mosaic_bound:      longint;
       show_imgs:         longint;
       dontuse_opencv:    longint;
       contrastive:       longint;
       contrastive_jit_flip: longint;
       contrastive_color: longint;
       jitter:            single;
       resize:            single;
       flip:              longint;
       gaussian_noise:    longint;
       blur:              longint;
       mixup:             longint;
      label_smooth_eps:   single;
      angle:              single;
      aspect:             single;
      saturation:         single;
      exposure:           single;
      hue:                single;
      d:                  PData;
      im:                 PImageData;
      resized:            PImageData;
      &type:              TDataType;
      hierarchy:          TArray<TTree>
    end;

  PBoxlabel = ^TBoxLabel;
  TBoxLabel = record
      id , track_id: longint;
      x : Single;
      y : Single;
      w : Single;
      h : Single;
      left : Single;
      right : Single;
      top : Single;
      bottom : Single;
    end;

  PNode = ^TNode;
  TNode = record
      val : pointer;
      next : PNode;
      prev : PNode;
    end;

  //PList = ^TList;
  //TList = record
  //    size : longint;
  //    front : PNode;
  //    back : PNode;
  //  end;

  PPNetwork = ^PNetwork;
  PNetwork = ^TNetwork;

  PLayer = ^TLayer;

  PLearningRatePolicy = ^TLearningRatePolicy;
  TLearningRatePolicy = (lrpCONSTANT, lrpSTEP, lrpEXP, lrpPOLY, lrpSTEPS, lrpSIG, lrpRANDOM, lrpSGDR);

{$ifdef GPU}

const
  //BLOCK = 512;
  //{$include cuda_runtime.inc}
  //{$include curand.inc}
  //{$include cublas_v2.inc}
  //{$ifdef CUDNN}
  //  {$include cudnn.inc}
  //{$endif}
{$endif}

  PNetworkState = ^TNetworkState;
  TNetworkState = record
      truth : TSingles;
      input : TSingles;
      delta : TSingles;
      workspace : TSingles;
      train : boolean;
      index : longint;
      net : PNetwork;
    end;


  TProcProgress = procedure(const progress: longint; const net :PNetwork);

  TMeasureOps = (opIncFill, opFill, opCopy, opNorm, opBatchAddvs, opAddvs, opBatchMulvs, opMulvs, opAxpy, opMulvv, opAddvv, opDot, opBatchFma, opFma, opGemm, opIm2col, opCol2im, opIm2ColExt, opCol2ImExt);

  { TMetrics }

  TMetrics = record
    type

      { TAct }

      TAct =record
      private
          m:array[0..999] of int64;
          stack: longint;
          function GetItem(i: TActivation): int64;
      public
          all:array[low(TActivation)..high(TActivation)] of int64;
          procedure start(const a:TActivation);
          procedure finish(const a:TActivation);
          function total:int64;
          property Item[i:TActivation]:int64 read GetItem ;default;
      end;

    { TFw }
       TFw = record
       private
          m:array[0..999] of int64;
          stack: longint;
          function GetItem(i: TLayerType): int64;
       public
          all:array[low(TLayerType)..high(TLayerType)] of int64;
          procedure start(const a:TLayerType);
          procedure finish(const a:TLayerType);
          function total():int64;
          property Item[i:TLayerType]:int64 read GetItem ;default;
       end;

       { TOps }

       TOps = record
       private
          m:array[0..999] of int64;
          stack: longint;
          function GetItem(i: TMeasureOps): int64;
       public

          all: array[low(TMeasureOps)..high(TMeasureOps)] of int64;
          procedure start(const a:TMeasureOps);
          procedure finish(const a:TMeasureOps);
          function total():int64;
          property Item[i:TMeasureOps]:int64 read GetItem ;default;
        end;
    public

      ops: TOps;
      act, grad : TAct;
      forward, backward:TFw;

      function print:string;
  end;

  { TLayer }

  TLayer = record
    type

      TPropagationProc = procedure (var layer:TLayer; const net:PNetworkState);
      TUpdateProc      = procedure (const layer:TLayer; const args:TUpdateArgs);

    public
      &type      : TLayerType;
      activation : TActivation;
      lstmActivation : TActivation;
      costType : TCostType;
      forward , backward : TPropagationProc;
      update : TUpdateProc;
{$ifdef GPU}
      forward_gpu, backward_gpu: TPropagationProc;
      update_gpu : TUpdateProc;
{$endif}

      share_layer : Player;
      train : boolean;
      avgpool : boolean;
      batch_normalize : boolean;
      shortcut : boolean;
      batch : longint;
      dynamic_minibatch : longint;
      forced : boolean;
      flipped : boolean;
      inputs : longint;
      outputs : longint;
      mean_alpha : single;
      nweights : longint;
      nbiases : longint;
      extra : longint;
      truths : longint;
      h : longint;
      w : longint;
      c : longint;
      out_h : longint;
      out_w : longint;
      out_c : longint;
      n : longint;
      max_boxes : longint;
      truth_size : longint;
      groups : longint;
      group_id : longint;
      size : longint;
      side : longint;
      stride : longint;
      stride_x : longint;
      stride_y : longint;
      dilation : longint;
      antialiasing : longint;
      maxpool_depth : longint;
      maxpool_zero_nonmax : longint;
      out_channels : longint;
      reverse : boolean;
      coordconv : longint;
      flatten : boolean;
      spatial : longint;
      pad : longint;
      sqrt : boolean;
      flip : boolean;
      index : longint;
      scale_wh : longint;
      binary : boolean;
      xnor : boolean;
      peephole : boolean;
      use_bin_output : boolean;
      keep_delta_gpu : boolean;
      optimized_memory : longint;
      steps : longint;
      history_size : longint;
      bottleneck : boolean;
      time_normalizer : single;
      state_constrain : longint;
      hidden : longint;
      truth : boolean;
      smooth : single;
      dot : single;
      deform : boolean;
      grad_centr : longint;
      sway : longint;
      rotate : longint;
      stretch : longint;
      stretch_sway : longint;
      angle : single;
      jitter : single;
      resize : single;
      saturation : single;
      exposure : single;
      shift : single;
      ratio : single;
      learning_rate_scale : single;
      clip : single;
      focal_loss : boolean;
      classes_multipliers : TArray<Single>;
      label_smooth_eps : single;
      noloss : boolean;
      softmax : boolean;
      classes : longint;
      detection : longint;
      embedding_layer_id : longint;
      embedding_output : TSingles;
      embedding_size : longint;
      sim_thresh : single;
      track_history_size : longint;
      dets_for_track : longint;
      dets_for_show : longint;
      track_ciou_norm : single;
      coords : longint;
      background : longint;
      rescore : boolean;
      objectness : longint;
      does_cost : longint;
      joint : longint;
      noadjust : boolean;
      reorg : longint;
      log : longint;
      tanh : boolean ;
      mask : TArray<longint>;
      total : longint;
      bflops : single;
      adam : longint;
      B1 : single;
      B2 : single;
      eps : single;
      t : longint;
      alpha : single;
      beta : single;
      kappa : single;
      coord_scale : single;
      object_scale : single;
      noobject_scale : single;
      mask_scale : single;
      class_scale : single;
      bias_match : boolean;
      random : boolean;
      ignore_thresh : single;
      truth_thresh : single;
      iou_thresh : single;
      thresh : single;
      focus : single;
      classfix : longint;
      absolute : longint;
      assisted_excitation : longint;
      onlyforward : boolean;
      stopbackward : boolean;
      train_only_bn : boolean;
      dont_update : boolean;
      burnin_update : boolean;
      dontload : boolean;
      dontsave : boolean;
      dontloadscales : boolean;
      numload : longint;
      temperature : single;
      probability : single;
      dropblock_size_rel : single;
      dropblock_size_abs : longint;
      dropblock : boolean;
      scale : single;
      receptive_w : longint;
      receptive_h : longint;
      receptive_w_scale : longint;
      receptive_h_scale : longint;
      cweights : TArray<byte>;//Pchar;
      indexes : TArray<longint>;//Plongint;
      input_layers : TArray<longint>;
      input_sizes : TArray<longint>;
      layers_output : TArray<TSingles>;//TSingles2D;
      layers_delta : TArray<TSingles>;//TSingles2D;
      weights_type : TWeightsType;
      weights_normalization : TWeightsNormalization;
      map : TArray<longint>;
      counts : TArray<longint>;
      sums : TSingles2D;
      rand : TSingles;
      cost : TSingles;
      labels : TArray<longint>;
      class_ids : TArray<longint>;
      contrastive_neg_max : longint;
      cos_sim : TSingles;
      exp_cos_sim : TSingles;
      p_constrastive : TSingles;
      contrast_p_gpu : PContrastiveParams;
      state : TSingles;
      prev_state : TSingles;
      forgot_state : TSingles;
      forgot_delta : TSingles;
      state_delta : TSingles;
      combine_cpu : TSingles;
      combine_delta_cpu : TSingles;
      concat : TSingles;
      concat_delta : TSingles;
      binary_weights : TSingles;
      biases : TSingles;
      bias_updates : TSingles;
      scales : TSingles;
      scale_updates : TSingles;
      weights_ema : TSingles;
      biases_ema : TSingles;
      scales_ema : TSingles;
      weights : TSingles;
      weight_updates : TSingles;
      scale_x_y : single;
      objectness_smooth : boolean;
      new_coords : boolean;
      show_details : longint;
      max_delta : single;
      uc_normalizer : single;
      iou_normalizer : single;
      obj_normalizer : single;
      cls_normalizer : single;
      delta_normalizer : single;
      iou_loss : TIOULoss;
      iou_thresh_kind : TIOULoss;
      nms_kind : TNMSKind;
      beta_nms : single;
      yolo_point : TYOLOPoint;
      align_bit_weights_gpu : PShortInt;
      mean_arr_gpu : TSingles;
      align_workspace_gpu : TSingles;
      transposed_align_workspace_gpu : TSingles;
      align_workspace_size : longint;
      align_bit_weights : TArray<ShortInt>;
      mean_arr : TSingles;
      align_bit_weights_size : longint;
      lda_align : longint;
      new_lda : longint;
      bit_align : longint;
      col_image : TSingles;
      delta : TSingles;
      output : TSingles;
      activation_input : TSingles;
      delta_pinned : longint;
      output_pinned : longint;
      loss : TSingles;
      squared : TSingles;
      norms : TSingles;
      spatial_mean : TSingles;
      mean : TSingles;
      variance : TSingles;
      mean_delta : TSingles;
      variance_delta : TSingles;
      rolling_mean : TSingles;
      rolling_variance : TSingles;
      // all layers batch normalization?
      x : TSingles;
      x_norm : TSingles;

      // Convolutionn layer : ADAM Optimizer
      m : TSingles;
      v : TSingles;
      bias_m : TSingles;
      bias_v : TSingles;
      scale_m : TSingles;
      scale_v : TSingles;

      // gru Layer
      z_cpu : TSingles;
      r_cpu : TSingles;
      //GRU + LSTM layer
      h_cpu : TSingles;

      // LSTM + CONVLSTM Layer
      prev_state_cpu : TSingles;
      temp_cpu : TSingles;
      temp2_cpu : TSingles;
      temp3_cpu : TSingles;
      dh_cpu : TSingles;
      hh_cpu : TSingles;
      prev_cell_cpu : TSingles;
      cell_cpu : TSingles;
      f_cpu : TSingles;
      i_cpu : TSingles;
      g_cpu : TSingles;
      o_cpu : TSingles;
      c_cpu : TSingles;
      // ConvLSTM layer
      stored_h_cpu : TSingles;
      stored_c_cpu : TSingles;
      dc_cpu : TSingles;
      binary_input : TSingles;
      bin_re_packed_input : PUInt32;
      t_bit_input : TArray<Byte>;

      // RNN and CRNN Layer
      input_layer : TArray<TLayer>;    // this exists also in Convolutionale + Max/Avg Pooling layers for antialiasing
      self_layer : TArray<TLayer>;
      output_layer : TArray<TLayer>;
      reset_layer : TArray<TLayer>;
      update_layer : TArray<TLayer>;
      state_layer : TArray<TLayer>;

      input_gate_layer : TArray<TLayer>;
      state_gate_layer : TArray<TLayer>;

      input_save_layer : TArray<TLayer>;
      state_save_layer : TArray<TLayer>;

      input_state_layer : TArray<TLayer>;
      state_state_layer : TArray<TLayer>;

      // GRU Layer
      //input_z_layer : TArray<TLayer>;
      //state_z_layer : TArray<TLayer>;
      //input_r_layer : TArray<TLayer>;
      //state_r_layer : TArray<TLayer>;
      //input_h_layer : TArray<TLayer>;
      //state_h_layer : TArray<TLayer>;
      wz : TArray<TLayer>;
      uz : TArray<TLayer>;
      wr : TArray<TLayer>;
      ur : TArray<TLayer>;
      wh : TArray<TLayer>;
      uh : TArray<TLayer>;

      // LSTM Layer + ConvLSTM
      uf : TArray<TLayer>;
      ui : TArray<TLayer>;
      ug : TArray<TLayer>;
      uo : TArray<TLayer>;

      wf : TArray<TLayer>;
      wg : TArray<TLayer>;
      wi : TArray<TLayer>;
      wo : TArray<TLayer>;

      //ConvLSTM Layer
      vf : TArray<TLayer>;
      vi : TArray<TLayer>;
      vo : TArray<TLayer>;

      softmax_tree : TArray<TTree>;
      workspace_size : size_t;
      indexes_gpu : Plongint;
      stream : longint;
      wait_stream_id : longint;
  {$ifdef GPU}
      z_gpu : TSingles;
      r_gpu : TSingles;
      h_gpu : TSingles;
      stored_h_gpu : TSingles;
      bottelneck_hi_gpu : TSingles;
      bottelneck_delta_gpu : TSingles;
      temp_gpu : TSingles;
      temp2_gpu : TSingles;
      temp3_gpu : TSingles;
      dh_gpu : TSingles;
      hh_gpu : TSingles;
      prev_cell_gpu : TSingles;
      prev_state_gpu : TSingles;
      last_prev_state_gpu : TSingles;
      last_prev_cell_gpu : TSingles;
      cell_gpu : TSingles;
      f_gpu : TSingles;
      i_gpu : TSingles;
      g_gpu : TSingles;
      o_gpu : TSingles;
      c_gpu : TSingles;
      stored_c_gpu : TSingles;
      dc_gpu : TSingles;
      m_gpu : TSingles;
      v_gpu : TSingles;
      bias_m_gpu : TSingles;
      scale_m_gpu : TSingles;
      bias_v_gpu : TSingles;
      scale_v_gpu : TSingles;
      combine_gpu : TSingles;
      combine_delta_gpu : TSingles;
      forgot_state_gpu : TSingles;
      forgot_delta_gpu : TSingles;
      state_gpu : TSingles;
      state_delta_gpu : TSingles;
      gate_gpu : TSingles;
      gate_delta_gpu : TSingles;
      save_gpu : TSingles;
      save_delta_gpu : TSingles;
      concat_gpu : TSingles;
      concat_delta_gpu : TSingles;
      binary_input_gpu : TSingles;
      binary_weights_gpu : TSingles;
      bin_conv_shortcut_in_gpu : TSingles;
      bin_conv_shortcut_out_gpu : TSingles;
      mean_gpu : TSingles;
      variance_gpu : TSingles;
      m_cbn_avg_gpu : TSingles;
      v_cbn_avg_gpu : TSingles;
      rolling_mean_gpu : TSingles;
      rolling_variance_gpu : TSingles;
      variance_delta_gpu : TSingles;
      mean_delta_gpu : TSingles;
      col_image_gpu : TSingles;
      x_gpu : TSingles;
      x_norm_gpu : TSingles;
      weights_gpu : TSingles;
      weight_updates_gpu : TSingles;
      weight_deform_gpu : TSingles;
      weight_change_gpu : TSingles;
      weights_gpu16 : TSingles;
      weight_updates_gpu16 : TSingles;
      biases_gpu : TSingles;
      bias_updates_gpu : TSingles;
      bias_change_gpu : TSingles;
      scales_gpu : TSingles;
      scale_updates_gpu : TSingles;
      scale_change_gpu : TSingles;
      input_antialiasing_gpu : TSingles;
      output_gpu : TSingles;
      output_avg_gpu : TSingles;
      activation_input_gpu : TSingles;
      loss_gpu : TSingles;
      delta_gpu : TSingles;
      cos_sim_gpu : TSingles;
      rand_gpu : TSingles;
      drop_blocks_scale : TSingles;
      drop_blocks_scale_gpu : TSingles;
      squared_gpu : TSingles;
      norms_gpu : TSingles;
      gt_gpu : TSingles;
      a_avg_gpu : TSingles;
      input_sizes_gpu : Plongint;
      layers_output_gpu : TSingles2D;
      layers_delta_gpu : TSingles2D;
    {$ifdef CUDNN}
      srcTensorDesc : cudnnTensorDescriptor_t;
      dstTensorDesc : cudnnTensorDescriptor_t;
      srcTensorDesc16 : cudnnTensorDescriptor_t;
      dstTensorDesc16 : cudnnTensorDescriptor_t;
      dsrcTensorDesc : cudnnTensorDescriptor_t;
      ddstTensorDesc : cudnnTensorDescriptor_t;
      dsrcTensorDesc16 : cudnnTensorDescriptor_t;
      ddstTensorDesc16 : cudnnTensorDescriptor_t;
      normTensorDesc : cudnnTensorDescriptor_t;
      normDstTensorDesc : cudnnTensorDescriptor_t;
      normDstTensorDescF16 : cudnnTensorDescriptor_t;
      weightDesc : cudnnFilterDescriptor_t;
      weightDesc16 : cudnnFilterDescriptor_t;
      dweightDesc : cudnnFilterDescriptor_t;
      dweightDesc16 : cudnnFilterDescriptor_t;
      convDesc : cudnnConvolutionDescriptor_t;
      fw_algo : cudnnConvolutionFwdAlgo_t;
      fw_algo16 : cudnnConvolutionFwdAlgo_t;
      bd_algo : cudnnConvolutionBwdDataAlgo_t;
      bd_algo16 : cudnnConvolutionBwdDataAlgo_t;
      bf_algo : cudnnConvolutionBwdFilterAlgo_t;
      bf_algo16 : cudnnConvolutionBwdFilterAlgo_t;
      poolingDesc : cudnnPoolingDescriptor_t;
    {$else}
      srcTensorDesc : pointer;
      dstTensorDesc : pointer;
      srcTensorDesc16 : pointer;
      dstTensorDesc16 : pointer;
      dsrcTensorDesc : pointer;
      ddstTensorDesc : pointer;
      dsrcTensorDesc16 : pointer;
      ddstTensorDesc16 : pointer;
      normTensorDesc : pointer;
      normDstTensorDesc : pointer;
      normDstTensorDescF16 : pointer;
      weightDesc : pointer;
      weightDesc16 : pointer;
      dweightDesc : pointer;
      dweightDesc16 : pointer;
      convDesc : pointer;
      fw_algo : TUNUSED_ENUM_TYPE;
      fw_algo16 : TUNUSED_ENUM_TYPE;
      bd_algo : TUNUSED_ENUM_TYPE;
      bd_algo16 : TUNUSED_ENUM_TYPE;
      bf_algo : TUNUSED_ENUM_TYPE;
      bf_algo16 : TUNUSED_ENUM_TYPE;
      poolingDesc : pointer;
    {$endif}
  {$endif}

      class operator initialize({$ifdef FPC}var{$else}out{$endif} o:TLayer);
      class operator finalize(var l:TLayer);
    end;

  TNetwork = record
    n : longint;
    batch : longint;
    seen : TArray<int64>;
    badlabels_reject_threshold : TArray<single>;
    delta_rolling_max : TArray<single>;
    delta_rolling_avg : TArray<single>;
    delta_rolling_std : TArray<single>;
    weights_reject_freq : longint;
    equidistant_point : longint;
    badlabels_rejection_percentage : single;
    num_sigmas_reject_badlabels : single;
    ema_alpha : single;
    cur_iteration : TArray<longint>;
    loss_scale : single;
    t : TArray<longint>;
    epoch : single;
    subdivisions : longint;
    layers : TArray<TLayer>;
    output : TSingles;
    policy : TLearningRatePolicy;
    //benchmark : boolean;
    total_bbox : TArray<longint>;
    rewritten_bbox : TArray<longint>;
    learning_rate : single;
    learning_rate_min : single;
    learning_rate_max : single;
    batches_per_cycle : longint;
    batches_cycle_mult : longint;
    momentum : single;
    decay : single;
    gamma : single;
    scale : single;
    power : single;
    time_steps : longint;
    step : longint;
    max_batches : longint;
    num_boxes : longint;
    train_images_num : longint;
    seq_scales : TArray<Single>;
    scales : TArray<Single>;
    steps : TArray<longint>;
    num_steps : longint;
    burn_in : longint;
    cudnn_half : longint;
    adam : boolean;
    B1 : single;
    B2 : single;
    eps : single;
    inputs : longint;
    outputs : longint;
    truths : longint;
    notruth : longint;
    h : longint;
    w : longint;
    c : longint;
    max_crop : longint;
    min_crop : longint;
    max_ratio : single;
    min_ratio : single;
    center : boolean;
    flip : longint;
    gaussian_noise : longint;
    blur : longint;
    mixup : longint;
    label_smooth_eps : single;
    resize_step : longint;
    attention : longint;
    adversarial : boolean;
    adversarial_lr : single;
    max_chart_loss : single;
    letter_box : longint;
    mosaic_bound : longint;
    contrastive : boolean;
    contrastive_jit_flip : boolean;
    contrastive_color : boolean;
    unsupervised : boolean;
    angle : single;
    aspect : single;
    exposure : single;
    saturation : single;
    hue : single;
    random : longint;
    track : longint;
    augment_speed : longint;
    sequential_subdivisions : longint;
    init_sequential_subdivisions : longint;
    current_subdivision : longint;
    try_fix_nan : longint;
    gpu_index : longint;
    hierarchy : TArray<TTree>;
    input : TSingles;
    truth : TSingles;
    delta : TSingles;
    workspace : TArray<Single>;
    train : boolean;
    index : longint;
    cost : TSingles;
    clip : single;
//{$ifdef GPU}
    delta_gpu : Psingle;
    output_gpu : Psingle;
    input_state_gpu : Psingle;
    input_pinned_cpu : Psingle;
    input_pinned_cpu_flag : longint;
    input_gpu : PPsingle;
    truth_gpu : PPsingle;
    input16_gpu : PPsingle;
    output16_gpu : PPsingle;
    max_input16_size : Psize_t;
    max_output16_size : Psize_t;
    wait_stream : longint;
    cuda_graph : pointer;
    cuda_graph_exec : pointer;
    use_cuda_graph : longint;
    cuda_graph_ready : Plongint;
    global_delta_gpu : Psingle;
    state_delta_gpu : Psingle;
    max_delta_gpu_size : size_t;
    optimized_memory : longint;
    dynamic_minibatch : longint;
    workspace_size_limit : size_t;
    onForward, onBackward :TProcProgress;
//{$endif}
  end;


{$ifdef API}

function get_metadata(&file:PChar):Tmetadata;                                                                                  winapi;external;
function read_tree(filename:PChar):Ptree;                                                                                      winapi;external;
procedure free_layer(para1:TLayer);                                                                                            winapi;external;

function load_network(cfg:PChar; weights:PChar; clear:longint):PNetwork;                                                       winapi;external;
function get_base_args(net:PNetwork):TLoadArgs;                                                                               winapi;external;
procedure free_data(d:TData);                                                                                                  winapi;external;

function load_data(args:TLoadArgs):pthread_t;                                                                                 winapi;external;
function read_data_cfg(filename:PChar):Plist;                                                                                  winapi;external;
function read_cfg(filename:PChar):Plist;                                                                                       winapi;external;
function read_file(filename:PChar):Pbyte;                                                                                      winapi;external;
function resize_data(orig:TData; w:longint; h:longint):TData;                                                                  winapi;external;
function tile_data(orig:TData; divs:longint; size:longint):PData;                                                              winapi;external;
function select_data(orig:PData; inds:PLongint):TData;                                                                         winapi;external;
procedure forward_network(net:PNetwork);                                                                                       winapi;external;
procedure backward_network(net:PNetwork);                                                                                      winapi;external;
procedure update_network(net:PNetwork);                                                                                        winapi;external;
function dot_cpu(N:longint; X:PSingle; INCX:longint; Y:PSingle; INCY:longint):Single;                                          winapi;external;
procedure axpy_cpu(N:longint; ALPHA:Single; X:PSingle; INCX:longint; Y:PSingle;  INCY:longint);                                winapi;external;
procedure copy_cpu(N:longint; X:PSingle; INCX:longint; Y:PSingle; INCY:longint);                                               winapi;external;
procedure scal_cpu(N:longint; ALPHA:Single; X:PSingle; INCX:longint);                                                          winapi;external;
procedure fill_cpu(N:longint; ALPHA:Single; X:PSingle; INCX:longint);                                                          winapi;external;
procedure normalize_cpu(x:PSingle; mean:PSingle; variance:PSingle; batch:longint; filters:longint; spatial:longint);           winapi;external;
procedure softmax(input:PSingle; n:longint; temp:Single; stride:longint; output:PSingle);                                      winapi;external;
function best_3d_shift_r(a:TImage; b:TImage; min:longint; max:longint):longint;                                                winapi;external;
{$ifdef GPU}
procedure axpy_gpu(N:longint; ALPHA:Single; X:PSingle; INCX:longint; Y:PSingle; 
            INCY:longint);                                                             winapi;external;
procedure fill_gpu(N:longint; ALPHA:Single; X:PSingle; INCX:longint);                  winapi;external;
procedure scal_gpu(N:longint; ALPHA:Single; X:PSingle; INCX:longint);                  winapi;external;
procedure copy_gpu(N:longint; X:PSingle; INCX:longint; Y:PSingle; INCY:longint);       winapi;external;
procedure cuda_set_device(n:longint);                                                  winapi;external;
procedure cuda_free(x_gpu:PSingle);                                                    winapi;external;
function cuda_make_array(x:PSingle; n:size_t):PSingle;                                 winapi;external;
procedure cuda_pull_array(x_gpu:PSingle; x:PSingle; n:size_t);                         winapi;external;
function cuda_mag_array(x_gpu:PSingle; n:size_t):Single;                               winapi;external;
procedure cuda_push_array(x_gpu:PSingle; x:PSingle; n:size_t);                         winapi;external;
procedure forward_network_gpu(net:PNetwork);                                           winapi;external;
procedure backward_network_gpu(net:PNetwork);                                          winapi;external;
procedure update_network_gpu(net:PNetwork);                                            winapi;external;
function train_networks(nets:PPnetwork; n:longint; d:TData; interval:longint):Single;  winapi;external;
procedure sync_nets(nets:PPnetwork; n:longint; interval:longint);                      winapi;external;
procedure harmless_update_network_gpu(net:PNetwork);                                   winapi;external;
{$endif}

function get_label(characters:PPImage; _string:PChar; size:longint):TImage;                                                       winapi;external;
procedure draw_label(a:TImage; r:longint; c:longint; &label:TImage; const rgb:PSingle);                                           winapi;external;
procedure save_Image(im:TImage; const name:PChar);                                                                                winapi;external;
procedure save_Image_options(im:TImage; const name:PChar; f:TIMTYPE; quality:longint);                                            winapi;external;
procedure get_next_batch(d:TData; n:longint; offset:longint; X:PSingle; y:PSingle);                                               winapi;external;
procedure grayscale_Image_3c(im:TImage);                                                                                          winapi;external;
procedure normalize_Image(p:TImage);                                                                                              winapi;external;
procedure matrix_to_csv(m:TMatrix);                                                                                               winapi;external;
function train_network_sgd(net:PNetwork; d:TData; n:longint):Single;                                                              winapi;external;
procedure rgbgr_Image(im:TImage);                                                                                                 winapi;external;
function copy_data(d:TData):TData;                                                                                                winapi;external;
function concat_data(d1:TData; d2:TData):TData;                                                                                   winapi;external;
function load_cifar10_data(filename:PChar):TData;                                                                                 winapi;external;
function matrix_topk_accuracy(truth:TMatrix; guess:TMatrix; k:longint):Single;                                                    winapi;external;
procedure matrix_add_matrix(from:TMatrix; &to:TMatrix);                                                                           winapi;external;
procedure scale_matrix(m:TMatrix; scale:Single);                                                                                  winapi;external;
function csv_to_matrix(filename:PChar):TMatrix;                                                                                   winapi;external;
function network_accuracies(net:PNetwork; d:TData; n:longint):PSingle;                                                            winapi;external;
function train_network_datum(net:PNetwork):Single;                                                                                winapi;external;
function make_random_Image(w:longint; h:longint; c:longint):TImage;                                                               winapi;external;
procedure denormalize_connected_layer(l:TLayer);                                                                                  winapi;external;
procedure denormalize_convolutional_layer(l:TLayer);                                                                              winapi;external;
procedure statistics_connected_layer(l:TLayer);                                                                                   winapi;external;
procedure rescale_weights(l:TLayer; scale:Single; trans:Single);                                                                  winapi;external;
procedure rgbgr_weights(l:TLayer);                                                                                                winapi;external;
function get_weights(l:TLayer):PImage;                                                                                            winapi;external;
procedure demo(cfgfile:PChar; weightfile:PChar; thresh:Single; cam_index:longint; const filename:PChar;
            names:PPChar; classes:longint; frame_skip:longint; prefix:PChar; avg:longint;
            hier_thresh:Single; w:longint; h:longint; fps:longint; fullscreen:longint);                                           winapi;external;
procedure get_detection_detections(l:TLayer; w:longint; h:longint; thresh:Single; dets:Pdetection);                               winapi;external;
function option_find_str(l:Plist; key:PChar; def:PChar):PChar;                                                                    winapi;external;
function option_find_int(l:Plist; key:PChar; def:longint):longint;                                                                winapi;external;
function option_find_int_quiet(l:Plist; key:PChar; def:longint):longint;                                                          winapi;external;
function parse_network_cfg(filename:PChar):PNetwork;                                                                              winapi;external;
procedure save_weights(net:PNetwork; filename:PChar);                                                                             winapi;external;
procedure load_weights(net:PNetwork; filename:PChar);                                                                             winapi;external;
procedure save_weights_upto(net:PNetwork; filename:PChar; cutoff:longint);                                                        winapi;external;
procedure load_weights_upto(net:PNetwork; filename:PChar; start:longint; cutoff:longint);                                         winapi;external;
procedure zero_objectness(l:TLayer);                                                                                              winapi;external;
procedure get_region_detections(l:TLayer; w:longint; h:longint; netw:longint; neth:longint;
            thresh:Single; map:PLongint; tree_thresh:Single; relative:longint; dets:Pdetection);                                  winapi;external;
function get_yolo_detections(l:TLayer; w:longint; h:longint; netw:longint; neth:longint;
           thresh:Single; map:PLongint; relative:longint; dets:Pdetection):longint;                                               winapi;external;
procedure free_network(net:PNetwork);                                                                                             winapi;external;
procedure set_batch_network(net:PNetwork; b:longint);                                                                             winapi;external;
procedure set_temp_network(net:PNetwork; t:Single);                                                                               winapi;external;
function load_Image(filename:PChar; w:longint; h:longint; c:longint):TImage;                                                      winapi;external;
function load_Image_color(filename:PChar; w:longint; h:longint):TImage;                                                           winapi;external;
function make_Image(w:longint; h:longint; c:longint):TImage;                                                                      winapi;external;
function resize_Image(im:TImage; w:longint; h:longint):TImage;                                                                    winapi;external;
procedure censor_Image(im:TImage; dx:longint; dy:longint; w:longint; h:longint);                                                  winapi;external;
function letterbox_Image(im:TImage; w:longint; h:longint):TImage;                                                                 winapi;external;
function crop_Image(im:TImage; dx:longint; dy:longint; w:longint; h:longint):TImage;                                              winapi;external;
function center_crop_Image(im:TImage; w:longint; h:longint):TImage;                                                               winapi;external;
function resize_min(im:TImage; min:longint):TImage;                                                                               winapi;external;
function resize_max(im:TImage; max:longint):TImage;                                                                               winapi;external;
function threshold_Image(im:TImage; thresh:Single):TImage;                                                                        winapi;external;
function mask_to_rgb(mask:TImage):TImage;                                                                                         winapi;external;
function resize_network(net:PNetwork; w:longint; h:longint):longint;                                                              winapi;external;
procedure free_matrix(m:TMatrix);                                                                                                 winapi;external;
procedure test_resize(filename:PChar);                                                                                            winapi;external;
function show_Image(p:TImage; const name:PChar; ms:longint):longint;                                                              winapi;external;
function copy_Image(p:TImage):TImage;                                                                                             winapi;external;
procedure draw_box_width(a:TImage; x1:longint; y1:longint; x2:longint; y2:longint; w:longint; r:Single; g:Single; b:Single);      winapi;external;
function get_current_rate(net:PNetwork):Single;                                                                                   winapi;external;
procedure composite_3d(f1:PChar; f2:PChar; &out:PChar; delta:longint);                                                            winapi;external;
function load_data_old(paths:PPChar; n:longint; m:longint; labels:PPChar; k:longint; w:longint; h:longint):TData;                 winapi;external;
function get_current_batch(net:PNetwork):size_t;                                                                                  winapi;external;
procedure constrain_Image(im:TImage);                                                                                             winapi;external;
function get_network_Image_layer(net:PNetwork; i:longint):TImage;                                                                 winapi;external;
function get_network_output_layer(net:PNetwork):TLayer;                                                                           winapi;external;
procedure top_predictions(net:PNetwork; n:longint; index:PLongint);                                                               winapi;external;
procedure flip_Image(a:TImage);                                                                                                   winapi;external;
function float_to_Image(w:longint; h:longint; c:longint; data:PSingle):TImage;                                                    winapi;external;
procedure ghost_Image(source:TImage; dest:TImage; dx:longint; dy:longint);                                                        winapi;external;
function network_accuracy(net:PNetwork; d:TData):Single;                                                                          winapi;external;
procedure random_distort_Image(im:TImage; hue:Single; saturation:Single; exposure:Single);                                        winapi;external;
procedure fill_Image(m:TImage; s:Single);                                                                                         winapi;external;
function grayscale_Image(im:TImage):TImage;                                                                                       winapi;external;
procedure rotate_Image_cw(im:TImage; times:longint);                                                                              winapi;external;
function what_time_is_it_now:double;                                                                                              winapi;external;
function rotate_Image(m:TImage; rad:Single):TImage;                                                                               winapi;external;
procedure visualize_network(net:PNetwork);                                                                                        winapi;external;
function box_iou(a:TBox; b:TBox):Single;                                                                                          winapi;external;
function load_all_cifar10:TData;                                                                                                  winapi;external;
function read_boxes(filename:PChar; n:PLongint):PBox_label;                                                                       winapi;external;
function float_to_box(f:PSingle; stride:longint):TBox;                                                                            winapi;external;
procedure draw_detections(im:TImage; dets:Pdetection; num:longint; thresh:Single; names:PPChar; alphabet:PPImage; classes:longint); winapi;external;
function network_predict_data(net:PNetwork; test:TData):TMatrix;                                                                  winapi;external;
function load_alphabet:PPImage;                                                                                                   winapi;external;
function get_network_Image(net:PNetwork):TImage;                                                                                  winapi;external;
function network_predict(net:PNetwork; input:PSingle):PSingle;                                                                    winapi;external;
function network_width(net:PNetwork):longint;                                                                                     winapi;external;
function network_height(net:PNetwork):longint;                                                                                    winapi;external;
function network_predict_Image(net:PNetwork; im:TImage):PSingle;                                                                  winapi;external;
procedure network_detect(net:PNetwork; im:TImage; thresh:Single; hier_thresh:Single; nms:Single;
            dets:Pdetection);                                                                                                     winapi;external;
function get_network_boxes(net:PNetwork; w:longint; h:longint; thresh:Single; hier:Single;
           map:PLongint; relative:longint; num:PLongint):Pdetection;                                                              winapi;external;
procedure free_detections(dets:Pdetection; n:longint);                                                                            winapi;external;
procedure reset_network_state(net:PNetwork; b:longint);                                                                           winapi;external;
function get_labels(filename:PChar):PPChar;                                                                                       winapi;external;
procedure do_nms_obj(dets:Pdetection; total:longint; classes:longint; thresh:Single);                                             winapi;external;
procedure do_nms_sort(dets:Pdetection; total:longint; classes:longint; thresh:Single);                                            winapi;external;
function make_matrix(rows:longint; cols:longint):TMatrix;                                                                         winapi;external;
{$ifdef OPENCV}
function open_video_stream(const f:PChar; c:longint; w:longint; h:longint; fps:longint):pointer;                                  winapi;external;
function get_Image_from_stream(p:pointer):TImage;                                                                                 winapi;external;
procedure make_window(name:PChar; w:longint; h:longint; fullscreen:longint);                                                      winapi;external;
{$endif}

procedure free_Image(m:TImage);                                                                                                   winapi;external;
function train_network(net:PNetwork; d:TData):Single;                                                                             winapi;external;
function load_data_in_thread(args:TLoadArgs):pthread_t;                                                                           winapi;external;
procedure load_data_blocking(args:TLoadArgs);                                                                                     winapi;external;
function get_paths(filename:PChar):Plist;                                                                                         winapi;external;
procedure hierarchy_predictions(predictions:PSingle; n:longint; hier:Ptree; only_leaves:longint; stride:longint);                 winapi;external;
procedure change_leaves(t:Ptree; leaf_list:PChar);                                                                                winapi;external;
function find_int_arg(argc:longint; argv:PPChar; arg:PChar; def:longint):longint;                                                 winapi;external;
function find_float_arg(argc:longint; argv:PPChar; arg:PChar; def:Single):Single;                                                 winapi;external;
function find_arg(argc:longint; argv:PPChar; arg:PChar):longint;                                                                  winapi;external;
function find_char_arg(argc:longint; argv:PPChar; arg:PChar; def:PChar):PChar;                                                    winapi;external;
function basecfg(cfgfile:PChar):PChar;                                                                                            winapi;external;
procedure find_replace(str:PChar; orig:PChar; rep:PChar; output:PChar);                                                           winapi;external;
procedure free_ptrs(ptrs:PPointer; n:longint);                                                                                    winapi;external;
function fgetl(fp:PFILE):PChar;                                                                                                   winapi;external;
procedure strip(s:PChar);                                                                                                         winapi;external;
function sec(clocks:clock_t):Single;                                                                                              winapi;external;
function list_to_array(l:Plist):PPointer;                                                                                         winapi;external;
procedure top_k(a:PSingle; n:longint; k:longint; index:PLongint);                                                                 winapi;external;
function read_map(filename:PChar):PLongint;                                                                                       winapi;external;
procedure error(const s:PChar);                                                                                                   winapi;external;
function max_index(a:PSingle; n:longint):longint;                                                                                 winapi;external;
function max_int_index(a:PLongint; n:longint):longint;                                                                            winapi;external;
function sample_array(a:PSingle; n:longint):longint;                                                                              winapi;external;
function random_index_order(min:longint; max:longint):PLongint;                                                                   winapi;external;
procedure free_list(l:Plist);                                                                                                     winapi;external;
function mse_array(a:PSingle; n:longint):Single;                                                                                  winapi;external;
function variance_array(a:PSingle; n:longint):Single;                                                                             winapi;external;
function mag_array(a:PSingle; n:longint):Single;                                                                                  winapi;external;
procedure scale_array(a:PSingle; n:longint; s:Single);                                                                            winapi;external;
function mean_array(a:PSingle; n:longint):Single;                                                                                 winapi;external;
function sum_array(a:PSingle; n:longint):Single;                                                                                  winapi;external;
procedure normalize_array(a:PSingle; n:longint);                                                                                  winapi;external;
function read_intlist(s:PChar; n:PLongint; d:longint):PLongint;                                                                   winapi;external;
function rand_size_t:size_t;                                                                                                      winapi;external;
function rand_normal:Single;                                                                                                      winapi;external;
function rand_uniform(min:Single; max:Single):Single;                                                                             winapi;external;
{$endif}

{$if defined(unix) or defined(posix)}       // should work on darwin too!
  {.$linklib c}

  clockid_t=longint;

  PTimeSpec=^TTimeSpec;
  TTimeSpec = record
    tv_sec: int64;
    tv_nsec: int64;
  end;
  const
  {$if defined(linux)}
   //posix timer
  CLOCK_REALTIME                  = 0;
  CLOCK_MONOTONIC                 = 1;
  CLOCK_PROCESS_CPUTIME_ID        = 2;
  CLOCK_THREAD_CPUTIME_ID         = 3;
  CLOCK_MONOTONIC_RAW             = 4;
  CLOCK_REALTIME_COARSE           = 5;
  CLOCK_MONOTONIC_COARSE          = 6;

  {$elseif defined(darwin)}
  CLOCK_REALTIME                  = 0;
  CLOCK_MONOTONIC_RAW             = 4;
  CLOCK_MONOTONIC_RAW_APPROX      = 5;
  CLOCK_MONOTONIC                 = 6;
  CLOCK_UPTIME_RAW                = 8;
  CLOCK_UPTIME_RAW_APPROX         = 9;
  CLOCK_PROCESS_CPUTIME_ID        = 12;
  CLOCK_THREAD_CPUTIME_ID         = 16;

  {$else}  // libc
   CLOCK_REALTIME           = 0;
   CLOCK_PROCESS_CPUTIME_ID = 2;
   CLOCK_THREAD_CPUTIME_ID  = 3;
   CLOCK_MONOTONIC_RAW      = 4;

  {$endif}
  THE_CLOCK=CLOCK_MONOTONIC_RAW;
  strTimeError = 'cannot read OS time!, ErrorNo [%s]';

  function clock_gettime(clk_id : clockid_t; tp: ptimespec) : longint  ;cdecl; external {$ifndef fpc}{$ifdef MACOS} 'libc.dylib' {$else}'libc.so'{$endif}{$endif};
{$endif}

function _inc(var i:longint):longint;inline;
function _cni(var i:longint):longint;inline;
function _dec(var i:longint):longint;inline;
function _ced(var i:longint):longint;inline;

function Sum(v:TSingles; const N:size_t):single;overload;
function SumOfSquares(const v:TSingles; const N:size_t):single; overload;
function MinValue(const v:TSingles; const N:size_t):single;overload;
function MaxValue(const v:TSingles; const N:size_t):single;overload;
function Mean(const v:TSingles; const N:size_t):single;overload;
function Variance(const v:TSingles; const N:size_t):Single; overload;
function StdDev(const v:TSingles; const N:size_t):Single; overload;
function Norm(const v:TSingles; const N:size_t):Single; overload;
function min(const a,b:int64):int64;overload;
function max(const a,b:int64):int64;overload;
function min(const a,b:single):single; overload;
function max(const a,b:single):single; overload;

{$ifndef FPC}
procedure FreeMemAndNil(var Mem);
{$endif}
function clock():clock_t;

// thread-safe randoms, must provide a seed ((ThreadID + GetTickCount) for example)
function trandom():double;                      inline; overload;
function trandom(const aMax:longword):longword; inline; overload;
procedure free_sublayer(var l: PLayer);
procedure free_layer(var l: TLayer);         overload;
procedure free_layer(const l: TArray<TLayer>); overload;
procedure free_layer_custom(var l: TLayer; const keep_cudnn_desc: boolean);

procedure readLayer(const p:PSingle; const N:longint);

{$ifdef MSWINDOWS}
function QueryPerformanceCounter(var lpPerformanceCount: Int64): longbool;stdcall; external 'kernel32.dll';
function QueryPerformanceFrequency(var lpFrequency: Int64): longbool;stdcall external 'kernel32.dll';
{$endif}
var
   metrics: TMetrics;
   ProcForwardProgress, ProcBackwardProgress : TProcProgress ;

implementation
uses

  cfg, blas
  ;




{$ifndef FPC}

procedure FreeMemAndNil(var Mem);
var P:PPtrInt;
begin
  p:=Pointer(Mem);
  Pointer(mem):=nil;
  Dec(P);
  FreeMem(P)
end;

{$endif}

function Sum(v:TSingles; const N:size_t):single;overload;
var i:size_t;
begin
  result := 0;
  for i:=0 to N-1 do
    result := result + v[i]
end;

function SumOfSquares(const v:TSingles; const N:size_t):single; overload;
var i:size_t;
begin
result :=0;
  for i:=0 to N-1 do
    result:= result + sqr(v[i])
end;

function MinValue(const v:TSingles; const N:size_t):single;overload;
var i:size_t;
begin
  result :=0;
  if N<=0 then exit;
  result := v[0];
  for i := 1 to N -1 do
    if v[i] < result then
      result := v[i];
end;

function MaxValue(const v:TSingles; const N:size_t):single;overload;
var i: size_t;
begin
  result :=0;
  if N<=0 then exit;
  result := v[0];
  for i := 1 to N -1 do
    if v[i] > result then
      result := v[i];
end;

function Mean(const v:TSingles; const N:size_t):single;overload;
begin
  result := Sum(v, N) / N
end;

function Variance(const v:TSingles; const N:size_t):Single; overload;
var m:single; i:size_t;
begin
  result :=0;
  if N<=0 then exit;
  m:=Mean(v, N);
  for i:=0 to N-1 do
    result:= result + sqr(v[i] - m);
  result := result / N
end;

function StdDev(const v:TSingles; const N:size_t):Single; overload;
begin
  result := Sqrt(Variance(v,N))
end;

function Norm(const v:TSingles; const N:size_t):Single; overload;
begin
  result := sqrt(SumOfSquares(v,N)/N)
end;

function min(const a,b:int64):int64;overload;
begin
  if b<a then exit(b);
  result :=a;
end;

function max(const a,b:int64):int64;overload;
begin
  if b>a then exit(b);
  result :=a;
end;

function min(const a,b:single):single; overload;
begin
  if b<a then exit(b);
  result :=a;
end;

function max(const a,b:single):single; overload;
begin
  if b>a then exit(b);
  result :=a;
end;


procedure free_sublayer(var l: PLayer);
begin
    if assigned(l) then
        begin
            free_layer(l[0]);
            FreeMemAndNil(l)
        end
end;

procedure free_layer(var l: TLayer);
begin
    free_layer_custom(l, false)
end;

procedure free_layer(const l: TArray<TLayer>);
var i:longint;
begin
    for i:=0 to length(l)-1 do
      free_layer_custom(l[i], false)
end;

procedure free_layer_custom(var l: TLayer; const keep_cudnn_desc: boolean);
begin
    if (l.share_layer <> nil) then           // don't free shared layers
        exit();
    if l.antialiasing<>0 then
        free_layer(l.input_layer);
    if l.&type = ltConvLSTM then
        begin
            if l.peephole then
                begin
                    free_layer(l.vf);
                    free_layer(l.vi);
                    free_layer(l.vo)
                end;
            //else
            //    begin
            //        free(l.vf);
            //        free(l.vi);
            //        free(l.vo)
            //    end;
            free_layer(l.wf);
            if not l.bottleneck then
                begin
                    free_layer(l.wi);
                    free_layer(l.wg);
                    free_layer(l.wo)
                end;
            free_layer(l.uf);
            free_layer(l.ui);
            free_layer(l.ug);
            free_layer(l.uo)
        end;
    if l.&type = ltCRNN then
        begin
            free_layer(l.input_layer);
            free_layer(l.self_layer);
            free_layer(l.output_layer);
            l.output := nil;
            l.delta := nil;
        {$ifdef GPU}
            l.output_gpu := nil;
            l.delta_gpu := nil
        {$endif}
        end;
    if l.&type = ltDROPOUT then
        begin
            if assigned(l.rand) then  l.rand.free;
        {$ifdef GPU}
            if l.rand_gpu then
                cuda_free(l.rand_gpu);
            if l.drop_blocks_scale then
                cuda_free_host(l.drop_blocks_scale);
            if l.drop_blocks_scale_gpu then
                cuda_free(l.drop_blocks_scale_gpu);
        {$endif}
            exit()
        end;
    //if assigned(l.mask) then l.mask.free;
    //if assigned(l.classes_multipliers) then l.classes_multipliers.free;
    //if assigned(l.cweights) then l.cweights.free;
    //if assigned(l.indexes) then l.indexes.free;
    //if assigned(l.input_layers) then l.input_layers.free;
    //if assigned(l.input_sizes) then l.input_sizes.free;
    //if assigned(l.layers_output) then l.layers_output.free;
    //if assigned(l.layers_delta) then l.layers_delta.free;
    //if assigned(l.map) then l.map.free;
    if assigned(l.rand) then l.rand.free;
    if assigned(l.cost) then l.cost.free;
    //if assigned(l.labels) and not l.detection then
    //    free(l.labels);
    //if assigned(l.class_ids) and not l.detection then
    //    free(l.class_ids);
    if assigned(l.cos_sim)          then l.cos_sim.free;
    if assigned(l.exp_cos_sim)      then l.exp_cos_sim.free;
    if assigned(l.p_constrastive)   then l.p_constrastive.free;
    if assigned(l.embedding_output) then l.embedding_output.free;
    if assigned(l.state)            then l.state.free;
    if assigned(l.prev_state)       then l.prev_state.free;
    if assigned(l.forgot_state)     then l.forgot_state.free;
    if assigned(l.forgot_delta)     then l.forgot_delta.free;
    if assigned(l.state_delta)      then l.state_delta.free;
    if assigned(l.concat)           then l.concat.free;
    if assigned(l.concat_delta)     then l.concat_delta.free;
    if assigned(l.binary_weights)   then l.binary_weights.free;
    if assigned(l.biases)           then l.biases.free;
    if assigned(l.bias_updates)     then l.bias_updates.free;
    if assigned(l.scales)           then l.scales.free;
    if assigned(l.scale_updates)    then l.scale_updates.free;
    if assigned(l.biases_ema)       then l.biases_ema.free;
    if assigned(l.scales_ema)       then l.scales_ema.free;
    if assigned(l.weights_ema)      then l.weights_ema.free;
    if assigned(l.weights)          then l.weights.free;
    if assigned(l.weight_updates)   then l.weight_updates.free;
    //if assigned(l.align_bit_weights) then l.align_bit_weights.free;
    if assigned(l.mean_arr)         then l.mean_arr.free;
{$ifdef GPU}
    if assigned(l.delta) and (l.delta_pinned<>0)   then
        begin
            cudaFreeHost(l.delta);
            l.delta := nil
        end;
    if assigned(l.output) and (l.output_pinned<>0) then
        begin
            cudaFreeHost(l.output);
            l.output := nil
        end;
{$endif}
    if assigned(l.delta)            then l.delta.free;
    if assigned(l.output)           then l.output.free;
    if assigned(l.activation_input) then l.activation_input.free;
    if assigned(l.squared)          then l.squared.free;
    if assigned(l.norms)            then l.norms.free;
    if assigned(l.spatial_mean)     then l.spatial_mean.free;
    if assigned(l.mean)             then l.mean.free;
    if assigned(l.variance)         then l.variance.free;
    if assigned(l.mean_delta)       then l.mean_delta.free;
    if assigned(l.variance_delta)   then l.variance_delta.free;
    if assigned(l.rolling_mean)     then l.rolling_mean.free;
    if assigned(l.rolling_variance) then l.rolling_variance.free;
    if assigned(l.x)                then l.x.free;
    if assigned(l.x_norm)           then l.x_norm.free;
    if assigned(l.m)                then l.m.free;
    if assigned(l.v)                then l.v.free;
    if assigned(l.z_cpu)            then l.z_cpu.free;
    if assigned(l.r_cpu)            then l.r_cpu.free;
    if assigned(l.binary_input)     then l.binary_input.free;
    if assigned(l.bin_re_packed_input) then FreeMemAndNil(l.bin_re_packed_input);
    //if assigned(l.t_bit_input)         then l.t_bit_input.free;
    if assigned(l.loss)                then l.loss.free;
    if assigned(l.f_cpu)               then l.f_cpu.free;
    if assigned(l.i_cpu)               then l.i_cpu.free;
    if assigned(l.g_cpu)               then l.g_cpu.free;
    if assigned(l.o_cpu)               then l.o_cpu.free;
    if assigned(l.c_cpu)               then l.c_cpu.free;
    if assigned(l.h_cpu)               then l.h_cpu.free;
    if assigned(l.temp_cpu)            then l.temp_cpu.free;
    if assigned(l.temp2_cpu)           then l.temp2_cpu.free;
    if assigned(l.temp3_cpu)           then l.temp3_cpu.free;
    if assigned(l.dc_cpu)              then l.dc_cpu.free;
    if assigned(l.dh_cpu)              then l.dh_cpu.free;
    if assigned(l.prev_state_cpu)      then l.prev_state_cpu.free;
    if assigned(l.prev_cell_cpu)       then l.prev_cell_cpu.free;
    if assigned(l.stored_c_cpu)        then l.stored_c_cpu.free;
    if assigned(l.stored_h_cpu)        then l.stored_h_cpu.free;
    if assigned(l.cell_cpu)            then l.cell_cpu.free;
{$ifdef GPU}
    if l.indexes_gpu then
        cuda_free(single(l.indexes_gpu));
    if l.contrast_p_gpu then
        cuda_free(single(l.contrast_p_gpu));
    if l.z_gpu then
        cuda_free(l.z_gpu);
    if l.r_gpu then
        cuda_free(l.r_gpu);
    if l.m_gpu then
        cuda_free(l.m_gpu);
    if l.v_gpu then
        cuda_free(l.v_gpu);
    if l.forgot_state_gpu then
        cuda_free(l.forgot_state_gpu);
    if l.forgot_delta_gpu then
        cuda_free(l.forgot_delta_gpu);
    if l.state_gpu then
        cuda_free(l.state_gpu);
    if l.state_delta_gpu then
        cuda_free(l.state_delta_gpu);
    if l.gate_gpu then
        cuda_free(l.gate_gpu);
    if l.gate_delta_gpu then
        cuda_free(l.gate_delta_gpu);
    if l.save_gpu then
        cuda_free(l.save_gpu);
    if l.save_delta_gpu then
        cuda_free(l.save_delta_gpu);
    if l.concat_gpu then
        cuda_free(l.concat_gpu);
    if l.concat_delta_gpu then
        cuda_free(l.concat_delta_gpu);
    if l.binary_input_gpu then
        cuda_free(l.binary_input_gpu);
    if l.binary_weights_gpu then
        cuda_free(l.binary_weights_gpu);
    if l.mean_gpu then
        cuda_free(l.mean_gpu).mean_gpu := nil;
    if l.variance_gpu then
        cuda_free(l.variance_gpu).variance_gpu := nil;
    if l.m_cbn_avg_gpu then
        cuda_free(l.m_cbn_avg_gpu).m_cbn_avg_gpu := nil;
    if l.v_cbn_avg_gpu then
        cuda_free(l.v_cbn_avg_gpu).v_cbn_avg_gpu := nil;
    if l.rolling_mean_gpu then
        cuda_free(l.rolling_mean_gpu).rolling_mean_gpu := nil;
    if l.rolling_variance_gpu then
        cuda_free(l.rolling_variance_gpu).rolling_variance_gpu := nil;
    if l.variance_delta_gpu then
        cuda_free(l.variance_delta_gpu).variance_delta_gpu := nil;
    if l.mean_delta_gpu then
        cuda_free(l.mean_delta_gpu).mean_delta_gpu := nil;
    if l.x_norm_gpu then
        cuda_free(l.x_norm_gpu);
    if l.gt_gpu then
        cuda_free(l.gt_gpu);
    if l.a_avg_gpu then
        cuda_free(l.a_avg_gpu);
    if l.align_bit_weights_gpu then
        cuda_free(single(l.align_bit_weights_gpu));
    if l.mean_arr_gpu then
        cuda_free(l.mean_arr_gpu);
    if l.align_workspace_gpu then
        cuda_free(l.align_workspace_gpu);
    if l.transposed_align_workspace_gpu then
        cuda_free(l.transposed_align_workspace_gpu);
    if l.weights_gpu then
        cuda_free(l.weights_gpu).weights_gpu := nil;
    if l.weight_updates_gpu then
        cuda_free(l.weight_updates_gpu).weight_updates_gpu := nil;
    if l.weight_deform_gpu then
        cuda_free(l.weight_deform_gpu).weight_deform_gpu := nil;
    if l.weights_gpu16 then
        cuda_free(l.weights_gpu16).weights_gpu16 := nil;
    if l.weight_updates_gpu16 then
        cuda_free(l.weight_updates_gpu16).weight_updates_gpu16 := nil;
    if l.biases_gpu then
        cuda_free(l.biases_gpu).biases_gpu := nil;
    if l.bias_updates_gpu then
        cuda_free(l.bias_updates_gpu).bias_updates_gpu := nil;
    if l.scales_gpu then
        cuda_free(l.scales_gpu).scales_gpu := nil;
    if l.scale_updates_gpu then
        cuda_free(l.scale_updates_gpu).scale_updates_gpu := nil;
    if l.input_antialiasing_gpu then
        cuda_free(l.input_antialiasing_gpu).input_antialiasing_gpu := nil;
    if l.optimized_memory < 2 then
        begin
            if l.x_gpu then
                cuda_free(l.x_gpu).x_gpu := nil;
            if l.output_gpu then
                cuda_free(l.output_gpu).output_gpu := nil;
            if l.output_avg_gpu then
                cuda_free(l.output_avg_gpu).output_avg_gpu := nil;
            if l.activation_input_gpu then
                cuda_free(l.activation_input_gpu).activation_input_gpu := nil
        end;
    if l.delta_gpu and ((l.optimized_memory < 1) or l.keep_delta_gpu and l.optimized_memory < 3) then
        cuda_free(l.delta_gpu).delta_gpu := nil;
    if l.cos_sim_gpu then
        cuda_free(l.cos_sim_gpu);
    if l.rand_gpu then
        cuda_free(l.rand_gpu);
    if l.squared_gpu then
        cuda_free(l.squared_gpu);
    if l.norms_gpu then
        cuda_free(l.norms_gpu);
    if l.input_sizes_gpu then
        cuda_free(single(l.input_sizes_gpu));
    if l.layers_output_gpu then
        cuda_free(single(l.layers_output_gpu));
    if l.layers_delta_gpu then
        cuda_free(single(l.layers_delta_gpu));
    if l.f_gpu then
        cuda_free(l.f_gpu);
    if l.i_gpu then
        cuda_free(l.i_gpu);
    if l.g_gpu then
        cuda_free(l.g_gpu);
    if l.o_gpu then
        cuda_free(l.o_gpu);
    if l.c_gpu then
        cuda_free(l.c_gpu);
    if l.h_gpu then
        cuda_free(l.h_gpu);
    if l.bottelneck_hi_gpu then
        cuda_free(l.bottelneck_hi_gpu);
    if l.bottelneck_delta_gpu then
        cuda_free(l.bottelneck_delta_gpu);
    if l.temp_gpu then
        cuda_free(l.temp_gpu);
    if l.temp2_gpu then
        cuda_free(l.temp2_gpu);
    if l.temp3_gpu then
        cuda_free(l.temp3_gpu);
    if l.dc_gpu then
        cuda_free(l.dc_gpu);
    if l.dh_gpu then
        cuda_free(l.dh_gpu);
    if l.prev_state_gpu then
        cuda_free(l.prev_state_gpu);
    if l.prev_cell_gpu then
        cuda_free(l.prev_cell_gpu);
    if l.stored_c_gpu then
        cuda_free(l.stored_c_gpu);
    if l.stored_h_gpu then
        cuda_free(l.stored_h_gpu);
    if l.last_prev_state_gpu then
        cuda_free(l.last_prev_state_gpu);
    if l.last_prev_cell_gpu then
        cuda_free(l.last_prev_cell_gpu);
    if l.cell_gpu then
        cuda_free(l.cell_gpu);
  {$ifdef CUDNN}
    if not keep_cudnn_desc then
        begin
            if l.srcTensorDesc then
                CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.srcTensorDesc));
            if l.dstTensorDesc then
                CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.dstTensorDesc));
            if l.srcTensorDesc16 then
                CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.srcTensorDesc16));
            if l.dstTensorDesc16 then
                CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.dstTensorDesc16));
            if l.dsrcTensorDesc then
                CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.dsrcTensorDesc));
            if l.ddstTensorDesc then
                CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.ddstTensorDesc));
            if l.dsrcTensorDesc16 then
                CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.dsrcTensorDesc16));
            if l.ddstTensorDesc16 then
                CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.ddstTensorDesc16));
            if l.normTensorDesc then
                CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.normTensorDesc));
            if l.normDstTensorDesc then
                CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.normDstTensorDesc));
            if l.normDstTensorDescF16 then
                CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.normDstTensorDescF16));
            if l.weightDesc then
                CHECK_CUDNN(cudnnDestroyFilterDescriptor(l.weightDesc));
            if l.weightDesc16 then
                CHECK_CUDNN(cudnnDestroyFilterDescriptor(l.weightDesc16));
            if l.dweightDesc then
                CHECK_CUDNN(cudnnDestroyFilterDescriptor(l.dweightDesc));
            if l.dweightDesc16 then
                CHECK_CUDNN(cudnnDestroyFilterDescriptor(l.dweightDesc16));
            if l.convDesc then
                CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(l.convDesc));
            if l.poolingDesc then
                CHECK_CUDNN(cudnnDestroyPoolingDescriptor(l.poolingDesc))
        end
  {$endif}
{$endif}
end;


procedure readLayer(const p: PSingle; const N: longint);
begin
  writeln(format('%d X mean[%.6f] variance[%.6f] min[%.6f] max[%.6f] mag[%.6f]',[ n, mean_array(p,n), variance_array(p,n), minValue(p,n), maxValue(p,n), sqrt(SumOfSquares(p,n))]));
  readln
end;

{$ifdef MSWINDOWS}
var
   CPUFreq: Int64;
   CPUFreqs : double;
{$endif}
function clock():clock_t;
{$ifdef MSWINDOWS}
{$else}
var
   TimeSpec : TTimeSpec;
{$endif}
begin
  // in NANO Seconds
  {$ifdef MSWINDOWS}
  //result:=GetTickCount64
  QueryPerformanceCounter(result);
  result := trunc(result / CPUFreqs)

  {$else}
  if clock_gettime(THE_CLOCK,@TimeSpec) <>0 then
      raise Exception.Createfmt(strTimeError,[SysErrorMessage(GetLastOSError)]);
  result:=TimeSpec.tv_sec*1000000000 + TimeSpec.tv_nsec;
  {$endif}
end;

function _inc(var i: longint): longint;
begin
  result := i;
  inc(i)
end;

function _cni(var i: longint): longint;
begin
  inc(i);
  result := i
end;

function _dec(var i: longint): longint;
begin
  result := i;
  dec(i)
end;

function _ced(var i: longint): longint;
begin
  dec(i);
  result := i
end;

function trandom( const aMax: longword): longword;
var seed:longword;
begin
  { derived from xorshift allogrith : https://en.wikipedia.org/wiki/Xorshift  }
  (* The state word must be initialized to non-zero *)
  (* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" *)
  seed := {1+ ThreadId * 5 +}  clock() mod MaxInt;
  seed := seed xor (seed shl 13);
  seed := seed xor (seed shr 17);
  seed := seed xor (seed shl 5);
  Result := seed mod aMax;
end;

function trandom(): double;
begin
  result := trandom(MaxInt)/MaxInt;
end;
{ TSinglesHelper }

constructor TSinglesHelper.Create(const aCount: size_t);
begin
  {$ifdef NO_POINTERS}
  setLength(self,aCount)
  {$else}
  {$ifdef FPC}
  self:=AllocMem(aCount*sizeOf(Single));
  {$else}
  var P:PPtrInt := AllocMem(aCount*sizeOf(Single)+sizeof(PtrInt));
  P^:=aCount;
  inc(P);
  self := Pointer(P);
  {$endif}
  {$endif}
end;

procedure TSinglesHelper.ReAllocate(const aCount: size_t);
var
    c,i:size_t;
begin
  {$ifdef NO_POINTERS}
  setLength(self,aCount)
  {$else}
  c:=count();
  {$ifndef FPC}
  var P :PPtrInt := PPtrInt(Self);
  dec(p);
  p:=ReallocMemory(p,aCount*sizeof(single)+sizeof(PtrInt));
  P^ := aCount;
  Inc(P);
  Self:= Pointer(P);
  {$else}
  Self:=ReallocMemory(self,aCount*sizeof(single));
  {$endif}
  if aCount>c then
    for i:=c to aCount-1 do
      self[i]:=default(single)
  {$endif}
end;

function TSinglesHelper.Count(): size_t;
begin
  {$ifdef NO_POINTERS}
  result:=length(Self);
  {$else}
  if not assigned(self) then exit(0);
  {$ifdef FPC}
    result:=MemSize(Self) div sizeof(Single);
  {$else}
    result:=PPtrInt(Self)[-1];
  {$endif}

  {$endif}
end;

function TSinglesHelper.High(): PtrInt;
begin
  {$ifdef NO_POINTERS}
  result:=high(Self)
  {$else}
  result:=PPtrInt(Self)[-1] -1;
  {$endif}
end;

procedure TSinglesHelper.Free;
begin
  {$ifndef NO_POINTERS}
  if assigned(self) then FreeMemAndNil(Self);
  {$endif}
end;

function TSinglesHelper.toString(const sep: string; N: longint): string;
var i:size_t;
    s:string;
begin
  result:='';
  if N<0 then N:=Count()
  else N:=Min(Count(), N);
  if not assigned(self) then exit;
  for i:=0 to N-1 do begin
    str(self[i]:3:2,s);
    result:=result+sep+s;
  end;
  delete(result,1,length(sep));
  result:=IntToStr(N)+' X [ '+result+' ]'
end;

{ TSingles2dHelper }

class function TSingles2dHelper.Create(const aCount: size_t): TSingles2d;
begin
  {$ifdef NO_POINTERS}
  setLength(self,aCount)
  {$else}

  {$ifdef FPC}
  result:=AllocMem(aCount*sizeOf(TSingles));
  {$else}
  var P:PPtrInt := AllocMem(aCount*sizeOf(TSingles)+sizeof(PtrInt));
  P^:=aCount;
  inc(P);
  Result := Pointer(P);
  {$endif}

  {$endif}
end;

procedure TSingles2dHelper.ReAllocate(const aCount: size_t);
var
  c,i:size_t;
begin
  {$ifdef NO_POINTERS}
  setLength(self,aCount)
  {$else}
  c:=count();
  {$ifndef FPC}
  var P :PPtrInt := PPtrInt(Self);
  dec(p);
  p:=ReallocMemory(p,aCount*sizeof(TSingles)+sizeof(PtrInt));
  P^ := aCount ;
  Inc(P);
  Self:= Pointer(P);
  {$else}
  self:=ReallocMemory(self,aCount*sizeof(TSingles));
  {$endif}
  if aCount>c then
    for i:=c to aCount-1 do
      self[i]:=default(TSingles)
  {$endif}
end;

function TSingles2dHelper.Count(): size_t;
begin
  {$ifdef NO_POINTERS}
  result:=length(Self);
  {$else}
  if not assigned(self) then exit(0);
  {$ifdef FPC}
  result:=MemSize(Self) div sizeof(TSingles);
  {$else}
  result :=PPtrInt(Self)[-1]
  {$endif}

  {$endif}
end;

function TSingles2dHelper.High(): PtrInt;
begin
  {$ifdef NO_POINTERS}
  result:=high(Self)
  {$else}
  result:=PPtrInt(Self)[-1] -1
  {$endif}
end;

procedure TSingles2dHelper.Free;
var i:longint;
begin
  {$ifndef NO_POINTERS}
  for i:=0 to count()-1 do
    Self[i].Free;
  if assigned(self) then FreeMemAndNil(Self);
  {$endif}
end;

function TSingles2dHelper.toString(const sep: string; N: longint): string;
var i:longint;
begin
  result:='';
  if N<0 then N:=Count()
  else N:=Min(Count(), N);
  for i:=0 to N-1 do
    result:=result+sep+self[i].toString;
  delete(result, 1, length(sep));
  result:=IntToStr(n)+' X [ '+result+' ]'
end;

{ TIntegersHelper }

constructor TIntegersHelper.Create(const aCount: size_t);
begin
  {$ifdef NO_POINTERS}
  setLength(self,aCount)
  {$else}
  {$ifdef FPC}
  self:=AllocMem(aCount*sizeOf(longint));
  {$else}
  var P:PPtrInt := AllocMem(aCount*sizeOf(longint)+sizeof(PtrInt));
  P^:=aCount;
  inc(P);
  self := Pointer(P);
  {$endif}

  {$endif}
end;

procedure TIntegersHelper.ReAllocate(const aCount: size_t);
var c,i:size_t;
begin
  {$ifdef NO_POINTERS}
  setLength(self,aCount)
  {$else}
  c:=count();
  {$ifndef FPC}
  var P :PPtrInt := PPtrInt(Self);
  dec(p);
  p:=ReallocMemory(p,aCount*sizeof(longint)+sizeof(PtrInt)+sizeof(PtrInt));
  P^ := aCount;
  Inc(P);
  Self:= Pointer(P);
  {$else}
  self:=ReallocMemory(self,aCount*sizeof(longint));
  {$endif}
  if aCount>c then
    for i:=c to aCount-1 do
      self[i] := default(longint)
  {$endif}
end;

function TIntegersHelper.Count(): size_t;
begin
  {$ifdef NO_POINTERS}
  result:=length(Self);
  {$else}
  if not assigned(self) then exit(0);
  {$ifdef FPC}
  result:=MemSize(Self) div sizeof(LongInt);
  {$else}
  result:=PPtrInt(Self)[-1]
  {$endif}
  {$endif}
end;

function TIntegersHelper.High(): PtrInt;
begin
  {$ifdef NO_POINTERS}
  result:=high(Self)
  {$else}
  result:=PPtrInt(Self)[-1] -1;
  {$endif}
end;

procedure TIntegersHelper.Free;
begin
  {$ifndef NO_POINTERS}
  if assigned(self) then
    FreeMemAndNil(Self);
  {$endif}
end;

function TIntegersHelper.toString(const sep:string): string;
var i:size_t;
    s:string;
begin
  result:='';
  for i:=0 to self.high do begin
    str(self[i],s);
    result:=result+sep+s;
  end;
  delete(result,1,length(sep));
  result:='[ '+result+' ]'
end;

{ TIntegers2dHelper }

class function TIntegers2dHelper.Create(const aCount: size_t): TIntegers2d;
begin
  {$ifdef NO_POINTERS}
  setLength(self,aCount)
  {$else}
  {$ifdef FPC}
  result:=AllocMem(aCount*sizeOf(TIntegers));
  {$else}
  var P:PPtrInt := AllocMem(aCount*sizeOf(TIntegers)+sizeof(PtrInt));
  P^:=aCount;
  inc(P);
  result := Pointer(P);
  {$endif}
  {$endif}
end;

procedure TIntegers2dHelper.ReAllocate(const aCount: size_t);
var c,i:size_t;
begin
  {$ifdef NO_POINTERS}
  setLength(self,aCount)
  {$else}
  c:=count();
  {$ifndef FPC}
  var P :PPtrInt := PPtrInt(Self);
  dec(p);
  p:=ReallocMemory(p,aCount*sizeof(TIntegers)+sizeof(PtrInt));
  P^ := aCount;
  Inc(P);
  Self:= Pointer(P);
  {$else}
  self:=ReallocMemory(self,aCount*sizeof(TIntegers));

  {$endif}
  if aCount>c then
    for i:=c to aCount-1 do
      self[i] := default(TIntegers)
  {$endif}
end;

function TIntegers2dHelper.Count(): size_t;
begin
  {$ifdef NO_POINTERS}
  result:=length(Self);
  {$else}
  if not assigned(self) then exit(0);
  {$ifdef FPC}
  result:=MemSize(Self) div sizeof(TIntegers);
  {$else}
  result:=PPtrInt(Self)[-1]
  {$endif}
  {$endif}
end;

function TIntegers2dHelper.High(): PtrInt;
begin
  {$ifdef NO_POINTERS}
  result:=high(Self)
  {$else}
  result:=PPtrInt(Self)[-1] -1
  {$endif}
end;

procedure TIntegers2dHelper.Free;
var i:longint;
begin
  {$ifndef NO_POINTERS}
  for i:=0 to count()-1 do
    Self[i].Free;
  if assigned(self) then FreeMemAndNil(Self);
  {$endif}
end;

function TIntegers2dHelper.toString(const sep: string): string;
var
  i: Integer;
begin
  result:='';
  for i:=0 to count()-1 do
    result:=result+sep+self[i].toString;
  delete(result, 1, length(sep)) ;
  result:='[ '+result+' ]'
end;

{ TMetrics }

function TMetrics.print: string;
const uSecPerSec=1000000;
var
  i :TMeasureOps;
  j :TActivation;
  k :TLayerType;
begin

  result:='';
  for i:= low(ops.all) to high(ops.all) do
    if ops.all[i]<>0 then
      result := result + format('%-15s%10.3f[ms]',[copy(GetEnumName(TypeInfo(TMeasureOps),ord(i)),3), ops.all[i]/uSecPerSec] ) + sLineBreak;
  result := result + '----------------------------' + sLineBreak;
  result := result + format('Total          %10.3f[ms]', [ops.total()/uSecPerSec]) + sLineBreak + sLineBreak;

  for j:= low(act.all) to high(act.all) do
    if act.all[j]<>0 then
      result := result + format('%-15s%10.3f[ms]',[copy(GetEnumName(TypeInfo(TActivation),ord(j)),3), act.all[j]/uSecPerSec] ) + sLineBreak;
  result := result + '----------------------------' + sLineBreak;
  result := result + format('Total          %10.3f[ms]', [act.total/uSecPerSec]) + sLineBreak + sLineBreak;

  for k:= low(forward.all) to high(forward.all) do
    if forward.all[k]<>0 then
      result := result + format('%-15s%10.3f[ms]',[copy(GetEnumName(TypeInfo(TLayerType),ord(k)),3), forward.all[k]/uSecPerSec] ) + sLineBreak;
  result := result + '----------------------------' + sLineBreak;
  result := result + format('Total          %10.3f[ms]', [forward.total/uSecPerSec]) + sLineBreak + sLineBreak;

end;

{ TMetrics.TAct }

function TMetrics.TAct.GetItem(i: TActivation): int64;
begin
  result := all[i]
end;

procedure TMetrics.TAct.start(const a: TActivation);
begin
  m[stack]:=clock;
  inc(stack)
  //all[a] := clock();
end;

procedure TMetrics.TAct.finish(const a: TActivation);
begin
  dec(stack);
  all[a] := all[a] + clock()- m[stack]
end;

function TMetrics.TAct.total: int64;
var
  i: TActivation;
begin
  result := 0;
  for i:=low(TActivation) to high(TActivation) do
    inc(result, all[i])
end;

{ TMetrics.TFw }

function TMetrics.TFw.GetItem(i: TLayerType): int64;
begin
  result := all[i];
end;

procedure TMetrics.TFw.start(const a: TLayerType);
begin
  m[stack]:=clock;
  inc(stack)
end;

procedure TMetrics.TFw.finish(const a: TLayerType);
begin
  dec(stack);
  all[a] := all[a] + clock()- m[stack]
end;

function TMetrics.TFw.total(): int64;
var
  i: TLayerType;
begin
  result := 0;
  for i:=low(TLayerType) to high(TLayerType) do
    inc(result, all[i])
end;

{ TMetrics.TOps }

function TMetrics.TOps.GetItem(i: TMeasureOps): int64;
begin
  result := all[i]
end;

procedure TMetrics.TOps.start(const a: TMeasureOps);
begin
  m[stack]:=clock;
  inc(stack)
end;

procedure TMetrics.TOps.finish(const a: TMeasureOps);
begin
  dec(stack);
  all[a] := all[a] + clock()- m[stack]
end;

function TMetrics.TOps.total(): int64;
var
  i: TMeasureOps;
begin
  result := 0;
  for i:=low(TMeasureOps) to high(TMeasureOps) do
    inc(result, all[i])
end;

{ TLayer }

class operator TLayer.initialize({$ifdef FPC}var{$else}out{$endif} o: TLayer);
begin

end;

class operator TLayer.finalize(var l: TLayer);
begin
  //free_layer(l)
end;

initialization
{$ifdef MSWINDOWS}
  QueryPerformanceFrequency(CPUFreq);
  CPUFreqs := CPUFreq / 1000000000;
{$endif}

end.

