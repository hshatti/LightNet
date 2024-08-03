unit blas;

{$ifdef fpc}
  {$mode delphi}
  {$ModeSwitch cvar}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
  {$ifdef CPUX86_64}
     {$asmmode intel}
  {$endif}
  {$ifdef MSWINDOWS}{$FPUType AVX2}{$endif}
{$else}
{$excessprecision off}
{$endif}
{$pointermath on}
{$WRITEABLECONST ON}

interface

uses
  math, sysutils, lightnet;

const _EPSILON = 0.000001;
      TWO_PI =6.2831853071795864769252866;

{$if not declared(TSingles)}
  type TSingles=PSingle;
{$endif}

{$if not declared(TSingles2D)}
  type TSingles2D=PPSingle;
{$endif}

function rand_scale(s: single):single;
function int_index(const a: TArray<longint>; const val, n: longint): longint;
function rand_int(min: longint; max: longint):longint;
function constrain(const _min, _max, a: single): single;
function constrain_int(a: longint; min: longint; max: longint):longint;
function dist_array(const a, b: TSingles; const n, sub: longint):single;
procedure translate_array(const a: PSingle; const n: longint; const s: single);
function mag_array(const a: PSingle; const n: longint):single;
procedure scale_array(const a: PSingle; const n: longint; const s: single);
procedure normalize_array(const a: TSingles; const n: longint);
function max_index(const a: PSingle; const n: longint): longint;       overload;
function min_index(const a: PSingle; const n: longint): longint;       overload;
function max_index(const a: TArray<Single>): longint; overload;
function min_index(const a: TArray<Single>): longint; overload;
function rand_normal():single;
function rand_uniform(const amin, amax: single):single;

procedure top_k(const a: TSingles; const n, k: longint; const index: PLongint);

function one_hot_encode(const a: TSingles; const n, k: longint): TSingles2d; overload;
procedure one_hot_encode(const a: TArray<single>; const n, k: longint ;var result:TArray<TArray<Single>>); overload;

function variance_array(const a:psingle; const n:longint):single;

function mse_array(const a: psingle; const n: integer): single;
procedure copy_cpu(const N: longint; const src:PSingle;const INCX:longint; const dst:PSingle; const INCY:longint);    overload;
//procedure copy_cpu(const N: longint; const X:TSingles;const INCX:longint; const Y:TSingles; const INCY:longint);  overload;

procedure add_bias(const output: PSingle; const biases: PSingle; const batch: longint; const n: longint; const size: longint);

procedure scale_bias(const output, scales:PSingle; const batch, n, size:longint);

procedure scale_add_bias(const output, scales, biases:PSingle; const batch, n, size:longint);

procedure scal_add_cpu(const N:longint; const ALPHA, BETA:single; const X:PSingle;const INCX:longint);

procedure reorg_cpu(const x: PSingle; const out_w, out_h, out_c, batch, stride: longint; const forward: boolean; const &out: PSingle);

procedure scal_cpu(const N:longint; const ALPHA: single; const X: PSingle; const INCX: longint);

procedure fill_cpu(const N:longint; const ALPHA:single; const X: PSingle; const INCX: longint);

procedure flatten(const x:PSingle; const size, layers, batch:longint; const forward:boolean);

function random_matrix(const rows, cols:longint):TSingles;

procedure axpy_cpu(const N: longint; const ALPHA:single; const X:PSingle; const INCX: longint; const Y: PSingle; const INCY:longint);

function dot_cpu(const N: longint; const X: PSIngle; const INCX:longint; const Y: PSingle; const INCY:longint):single;

procedure test_blas;

procedure inter_cpu(const NX:longint; const X:PSingle; const NY:longint; const Y:PSingle; const B:longint; const &out:PSingle);

procedure deinter_cpu(const NX:longint; const X:PSingle; const NY:longint; const Y:PSingle; const B:longint; const &out:PSingle);

procedure mult_add_into_cpu(const N:longint; const X, Y, Z:PSingle);

procedure const_cpu(const N:longint; const ALPHA:single; const X:PSingle; const INCX:longint);


function sum_array(const a:PSingle; const n:longint):single;

function mean_array(const a:PSingle; const n:longint):single;

procedure pow_cpu(const N:longint; const ALPHA:single; const X:PSingle; const INCX:longint; const Y:PSingle; const INCY:longint);

procedure mul_cpu(const N:longint; const X:PSingle; const INCX:longint; const Y:PSingle; const INCY:longint);

procedure backward_bias(const bias_updates: PSingle; const delta: PSingle; const batch: longint; const n: longint; const size: longint);

procedure shortcut_cpu(const batch, w1, h1, c1:longint; const add:PSingle; const w2, h2, c2:longint; {const s1, s2:single;} const &out:PSingle);

procedure weighted_sum_cpu(const a, b, s:PSingle; const n:longint; const c:PSingle);

procedure weighted_delta_cpu(const a, b, s, da, db, ds:PSingle; const n:longint; const dc:PSingle);

procedure shortcut_multilayer_cpu(size: longint; src_outputs: longint; batch: longint; n: longint; outputs_of_layers: TArray<longint>; layers_output: PPsingle; &out: Psingle; &in: Psingle; weights: Psingle; nweights: longint; weights_normalization: TWeightsNormalization);

procedure backward_shortcut_multilayer_cpu(size: longint; src_outputs: longint; batch: longint; n: longint; outputs_of_layers: TArray<longint>; layers_delta: PPsingle; delta_out: Psingle; delta_in: Psingle; weights: Psingle; weight_updates: Psingle; nweights: longint; &in: Psingle; layers_output: PPsingle; weights_normalization: TWeightsNormalization);

procedure backward_scale_cpu(const x_norm, delta:PSingle; const batch, n, size:longint; const scale_updates:PSingle);

procedure mean_cpu(const x:PSingle; const batch, filters, spatial:longint; const mean:PSingle);

procedure variance_cpu(const x, mean:PSingle; const batch, filters, spatial:longint; const variance:PSingle);


procedure mean_delta_cpu(const delta, variance:PSingle; const batch, filters, spatial:longint; const mean_delta:PSingle);

procedure variance_delta_cpu(const x, delta, mean, variance:PSingle; const batch, filters, spatial:longint; const variance_delta:PSingle);

procedure normalize_delta_cpu(const x, mean, variance, mean_delta, variance_delta:PSingle; const batch, filters, spatial:longint; const delta:PSingle);

procedure normalize_cpu(const x, mean, variance: PSingle; const batch, filters, spatial: longint);

procedure l2normalize_cpu(const x, dx:PSingle; const batch, filters, spatial:longint);

procedure smooth_l1_cpu(const n:longint; const pred, truth, delta, error:PSingle);

procedure l2_cpu(const n:longint; const pred, truth, delta, error:PSingle);

procedure l1_cpu(const n:longint; const pred, truth, delta, error:PSingle);

procedure logistic_x_ent_cpu(const n:longint; const pred, truth, delta, error:PSingle);

procedure softmax_x_ent_cpu(const n: longint; const pred, truth, delta, error: PSingle);

procedure softmax(const input:PSingle; const n: longint; const temp: single; const stride: longint; const output:PSingle); overload;
procedure softmax(const input:PSingle; const n: longint; const temp: single; const output:PSingle; const stride: longint); overload;

procedure softmax_cpu(const input:PSingle; const n, batch, batch_offset, groups, group_offset, stride:longint; const temp:single; const output:PSingle);

procedure upsample_cpu(const &in:PSingle; const w, h, c, batch, stride:longint; const forward:boolean; const scale:single; const &out:PSingle);

procedure transpose_matrix(const a:TSingles; const rows, cols: integer);

procedure constrain_cpu(size: longint; ALPHA: single; X: Psingle);
procedure fix_nan_and_inf_cpu(input: Psingle; size: size_t);
procedure get_embedding(src: Psingle; src_w: longint; src_h: longint; src_c: longint; embedding_size: longint; cur_w: longint; cur_h: longint; cur_n: longint; cur_b: longint; dst: Psingle);
function math_vector_length(A: Psingle; feature_size: Longword):single;
function cosine_similarity(A: Psingle; B: Psingle; feature_size: Longword):single;
function get_sim_P_index(i: size_t; j: size_t; contrast_p: PContrastiveParams; contrast_p_size: longint):longint;
function check_sim(i: size_t; j: size_t; contrast_p: PContrastiveParams; contrast_p_size: longint):longint;
function find_sim(i: size_t; j: size_t; contrast_p: PContrastiveParams; contrast_p_size: longint):single;
function find_P_constrastive(i: size_t; j: size_t; contrast_p: PContrastiveParams; contrast_p_size: longint):single;
function P_constrastive_f_det(il: size_t; labels: Plongint; z: PPsingle; feature_size: Longword; temperature: single; contrast_p: PContrastiveParams; contrast_p_size: longint):single;
function P_constrastive_f(i: size_t; l: size_t; labels: Plongint; z: PPsingle; feature_size: Longword; temperature: single; contrast_p: PContrastiveParams; contrast_p_size: longint):single;
procedure grad_contrastive_loss_positive_f(i: size_t; class_ids: Plongint; labels: Plongint; num_of_samples: size_t; z: PPsingle; feature_size: Longword; temperature: single; delta: Psingle; wh: longint; contrast_p: PContrastiveParams; contrast_p_size: longint);
procedure grad_contrastive_loss_negative_f(i: size_t; class_ids: Plongint; labels: Plongint; num_of_samples: size_t; z: PPsingle; feature_size: Longword; temperature: single; delta: Psingle; wh: longint; contrast_p: PContrastiveParams; contrast_p_size: longint; neg_max: longint);
function P_constrastive(i: size_t; l: size_t; labels: Plongint; num_of_samples: size_t; z: PPsingle; feature_size: Longword; temperature: single; cos_sim: Psingle; exp_cos_sim: Psingle):single;
procedure grad_contrastive_loss_positive(i: size_t; labels: Plongint; num_of_samples: size_t; z: PPsingle; feature_size: Longword; temperature: single; cos_sim: Psingle; p_constrastive: Psingle; delta: Psingle; wh: longint);
procedure grad_contrastive_loss_negative(i: size_t; labels: Plongint; num_of_samples: size_t; z: PPsingle; feature_size: Longword; temperature: single; cos_sim: Psingle; p_constrastive: Psingle; delta: Psingle; wh: longint);
{$if defined(CPUX64) and defined(FPUAVX2)}
procedure smulvs(const dst, src:PSingle; const scale:single; const N:longint);assembler;
procedure smulvv(const dst, src, src2:PSingle; const N:longint);assembler;
procedure saddvs(const dst, src:PSingle; const s:single; const N:longint);assembler;
procedure saddvv(const dst, src, src2:PSingle; const N:longint);assembler;
{$endif}

{$ifndef fpc}
procedure filldword(var X; const N:PtrInt; const a:longword);
{$endif}

{$ifdef GPU}
{$include cuda.inc}
{$include tree.inc}

procedure constrain_gpu(N:longint; ALPHA:single; X:PSingle; INCX:longint);

function test_gpu_blas:longint;

procedure axpy_gpu(N:longint; ALPHA:single; X:PSingle; INCX:longint; Y:PSingle;
            INCY:longint);

procedure axpy_gpu_offset(N:longint; ALPHA:single; X:PSingle; OFFX:longint; INCX:longint;
            Y:PSingle; OFFY:longint; INCY:longint);

procedure copy_gpu(N:longint; X:PSingle; INCX:longint; Y:PSingle; INCY:longint);

procedure copy_gpu_offset(N:longint; X:PSingle; OFFX:longint; INCX:longint; Y:PSingle;
            OFFY:longint; INCY:longint);

procedure add_gpu(N:longint; ALPHA:single; X:PSingle; INCX:longint);

procedure supp_gpu(N:longint; ALPHA:single; X:PSingle; INCX:longint);

procedure mask_gpu(N:longint; X:PSingle; mask_num:single; mask:PSingle; val:single);

procedure scale_mask_gpu(N:longint; X:PSingle; mask_num:single; mask:PSingle; scale:single);

procedure const_gpu(N:longint; ALPHA:single; X:PSingle; INCX:longint);

procedure pow_gpu(N:longint; ALPHA:single; X:PSingle; INCX:longint; Y:PSingle;
            INCY:longint);

procedure mul_gpu(N:longint; X:PSingle; INCX:longint; Y:PSingle; INCY:longint);

procedure mean_gpu(x:PSingle; batch:longint; filters:longint; spatial:longint; mean:PSingle);

procedure variance_gpu(x:PSingle; mean:PSingle; batch:longint; filters:longint; spatial:longint;
            variance:PSingle);

procedure normalize_gpu(x:PSingle; mean:PSingle; variance:PSingle; batch:longint; filters:longint;
            spatial:longint);

procedure l2normalize_gpu(x:PSingle; dx:PSingle; batch:longint; filters:longint; spatial:longint);

procedure normalize_delta_gpu(x:PSingle; mean:PSingle; variance:PSingle; mean_delta:PSingle; variance_delta:PSingle;
            batch:longint; filters:longint; spatial:longint; delta:PSingle);

procedure fast_mean_delta_gpu(delta:PSingle; variance:PSingle; batch:longint; filters:longint; spatial:longint;
            mean_delta:PSingle);

procedure fast_variance_delta_gpu(x:PSingle; delta:PSingle; mean:PSingle; variance:PSingle; batch:longint;
            filters:longint; spatial:longint; variance_delta:PSingle);

procedure fast_variance_gpu(x:PSingle; mean:PSingle; batch:longint; filters:longint; spatial:longint;
            variance:PSingle);

procedure fast_mean_gpu(x:PSingle; batch:longint; filters:longint; spatial:longint; mean:PSingle);

procedure shortcut_gpu(batch:longint; w1:longint; h1:longint; c1:longint; add:PSingle;
            w2:longint; h2:longint; c2:longint; s1:single; s2:single;
            &out:PSingle);

procedure scale_bias_gpu(output:PSingle; biases:PSingle; batch:longint; n:longint; size:longint);

procedure backward_scale_gpu(x_norm:PSingle; delta:PSingle; batch:longint; n:longint; size:longint;
            scale_updates:PSingle);

procedure scale_bias_gpu(output:PSingle; biases:PSingle; batch:longint; n:longint; size:longint);

procedure add_bias_gpu(output:PSingle; biases:PSingle; batch:longint; n:longint; size:longint);

procedure backward_bias_gpu(bias_updates:PSingle; delta:PSingle; batch:longint; n:longint; size:longint);

procedure logistic_x_ent_gpu(n:longint; pred:PSingle; truth:PSingle; delta:PSingle; error:PSingle);

procedure softmax_x_ent_gpu(n:longint; pred:PSingle; truth:PSingle; delta:PSingle; error:PSingle);

procedure smooth_l1_gpu(n:longint; pred:PSingle; truth:PSingle; delta:PSingle; error:PSingle);

procedure l2_gpu(n:longint; pred:PSingle; truth:PSingle; delta:PSingle; error:PSingle);

procedure l1_gpu(n:longint; pred:PSingle; truth:PSingle; delta:PSingle; error:PSingle);

procedure wgan_gpu(n:longint; pred:PSingle; truth:PSingle; delta:PSingle; error:PSingle);

procedure weighted_delta_gpu(a:PSingle; b:PSingle; s:PSingle; da:PSingle; db:PSingle;
            ds:PSingle; num:longint; dc:PSingle);

procedure weighted_sum_gpu(a:PSingle; b:PSingle; s:PSingle; num:longint; c:PSingle);

procedure mult_add_into_gpu(num:longint; a:PSingle; b:PSingle; c:PSingle);

procedure inter_gpu(NX:longint; X:PSingle; NY:longint; Y:PSingle; B:longint;
            &out:PSingle);

procedure deinter_gpu(NX:longint; X:PSingle; NY:longint; Y:PSingle; B:longint;
            &out:PSingle);

procedure reorg_gpu(x:PSingle; w:longint; h:longint; c:longint; batch:longint;
            stride:longint; forward:longint; &out:PSingle);

procedure softmax_gpu(input:PSingle; n:longint; batch:longint; batch_offset:longint; groups:longint;
            group_offset:longint; stride:longint; temp:single; output:PSingle);

procedure adam_update_gpu(w:PSingle; d:PSingle; m:PSingle; v:PSingle; B1:single;
            B2:single; eps:single; decay:single; rate:single; n:longint;
            batch:longint; t:longint);

procedure adam_gpu(n:longint; x:PSingle; m:PSingle; v:PSingle; B1:single;
            B2:single; rate:single; eps:single; t:longint);

procedure flatten_gpu(x:PSingle; spatial:longint; layers:longint; batch:longint; forward:longint;
            &out:PSingle);

procedure softmax_tree(input:PSingle; spatial:longint; batch:longint; stride:longint; temp:single;
            output:PSingle; hier:tree);

procedure upsample_gpu(&in:PSingle; w:longint; h:longint; c:longint; batch:longint;
            stride:longint; forward:longint; scale:single; &out:PSingle);

{$endif}


implementation
uses steroids;
// todo SIMDfy blas
// todo make option to use openblas or intel MKL

{$if defined(CPUX64) and defined(FPUAVX2)}
procedure smulvs(const dst, src: PSingle; const scale: single; const N: longint);assembler;
{$ifdef FPC}nostackframe;{$endif}
asm
  {$ifndef FPC}
  .NOFRAME
  {$endif}
    vbroadcastss ymm1      , scale

    mov          r11d      , N
    shr          r11       , 3             // div regs
    jz           @rem

@while1:

    vmulps       ymm0      , ymm1   ,  [src]
    vmovups      [dst]     , ymm0

    add          src       , 32
    add          dst       , 32
    dec          r11
    jz           @rem
    jmp          @while1              // n<0

@rem:
    //mov          r11       , N
    and          N         , 7       // mod regs
    jz           @done

@while2:
    vmulss       xmm0      , xmm1   ,  dword ptr [src]
    vmovss       dword[dst]     , xmm0
    add          src       , 4
    add          dst       , 4

    dec          N
    jnz          @while2

@done:
end;

procedure smulvv(const dst, src, src2: PSingle; const N: longint);assembler;
{$ifdef FPC}nostackframe;{$endif}
asm
{$ifndef FPC}
.NOFRAME
{$endif}

  mov          r11d      , N
  shr          r11       , 3             // div regs
  jz           @rem

@while1:
  vmovups      ymm1      , [src2]
  vmulps       ymm0      , ymm1   ,  [src]
  vmovups      [dst]     , ymm0

  add          src2      , 32
  add          src       , 32
  add          dst       , 32
  dec          r11
  jz           @rem
  jmp          @while1              // n<0

@rem:
  //mov          r11       , N
  and          N         , 7       // mod regs
  jz           @done

@while2:
  vmovss       xmm1      , dword [src2]
  vmulss       xmm0      , xmm1   ,  dword ptr [src]
  vmovss       dword [dst]     , xmm0
  add          src2      , 4
  add          src       , 4
  add          dst       , 4

  dec          N
  jnz          @while2

@done:
end;

procedure saddvs(const dst, src: PSingle; const s: single; const N: longint);assembler;
{$ifdef FPC}nostackframe;{$endif}
asm
  {$ifndef FPC}
  .NOFRAME
  {$endif}
    vbroadcastss ymm1      , s

    mov          r11d      , N
    shr          r11       , 3             // div regs
    jz           @rem

@while1:

    vaddps       ymm0      , ymm1   ,  [src]
    vmovups      [dst]     , ymm0

    add          src         , 32
    add          dst         , 32
    dec          r11
    jnz          @while1

@rem:
  //mov          r11       , N
    and          N         , 7       // mod regs
    jz           @done

@while2:
    vaddss       xmm0      , xmm1   ,  dword ptr [src]
    vmovss       dword [dst]     , xmm0
    add          src       , 4
    add          dst       , 4

    dec          N
    jnz          @while2

@done:

end;

procedure saddvv(const dst, src, src2: PSingle; const N: longint);assembler;
{$ifdef FPC}nostackframe;{$endif}
asm
{$ifndef FPC}
.NOFRAME
{$endif}

  mov          r11d      , N
  shr          r11       , 3             // div regs
  jz           @rem

@while1:
  vmovups      ymm1      , [src2]
  vaddps       ymm0      , ymm1   ,  [src]
  vmovups      [dst]     , ymm0

  add          src2      , 32
  add          src       , 32
  add          dst       , 32
  dec          r11
  jz           @rem
  jmp          @while1              // n<0

@rem:
  //mov          r11       , N
  and          N         , 7       // mod regs
  jz           @done

@while2:
  vmovss       xmm1      , dword [src2]
  vaddss       xmm0      , xmm1   ,  dword ptr [src]
  vmovss       dword [dst]     , xmm0
  add          src2      , 4
  add          src       , 4
  add          dst       , 4

  dec          N
  jnz          @while2

@done:
end;

{$endif}


function rand_scale(s: single):single;
var
    scale: single;
begin
    scale := rand_uniform(1, s);
    if boolean(trandom(2)) then
        exit(scale);
    exit(1.0 / scale)
end;

function int_index(const a: TArray<longint>; const val, n: longint): longint;
var i:longint;
begin
  for i:=0 to n-1 do
      if a[i]=val then exit(i);
  result := -1
end;

function rand_int(min: longint; max: longint):longint;
var
    s: longint;
    r: longint;
begin
    if max < min then
        begin
            s := min;
            min := max;
            max := s
        end;
    r := trandom(max-min+1)+min;
    exit(r)
end;


function constrain(const _min, _max, a: single): single;
begin
  if a>_max then exit(_max);
  if a<_max then exit(_min);
  result:=a;
end;

function constrain_int(a: longint; min: longint; max: longint):longint;
begin
    if (a < min) then
        exit(min);
    if a > max then
        exit(max);
    exit(a)
end;

function dist_array(const a, b: TSingles; const n, sub: longint):single;
var
    i: longint;
    sum: single;
begin
    sum := 0;
    i := 0;
    while i < n do begin
        sum := sum + sqr(a[i]-b[i]{, 2});
        i := i + sub
    end;
    exit(sqrt(sum))
end;


procedure translate_array(const a: PSingle; const n: longint; const s: single);
var
    i: longint;
begin
  // todo SIMDify translate_array
  {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.start(opAddvs);{$endif}
  for i := 0 to n -1 do
        a[i] := a[i] + s ;
  {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.finish(opAddvs);{$endif}
end;

function mag_array(const a: PSingle; const n: longint):single;
var
    i: longint;
begin
  // todo SIMDify mag_array (also refactor with a sqr_array)
  result := 0;
  for i := 0 to n -1 do
      result := result + (a[i] * a[i]);
  result := sqrt(result)
end;

procedure scale_array(const a: PSingle; const n: longint; const s: single);
var
    i: longint;
begin
    {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.start(opMulvs);{$endif}
    {$if defined(CPUX64) and defined(AVX2)}
    smulvs(a,a,s,n);
    {$else}
    for i := 0 to n -1 do
        a[i] := a[i] * s;
    {$endif}
    {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.finish(opMulvs);{$endif}
end;

procedure normalize_array(const a: TSingles; const n: longint);
var i:integer;
    mu, sigma:single;
begin
  {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.start(opNorm);{$endif}
  mu := mean_array(a,n);
  sigma := sqrt(variance_array(a,n));
  for i := 0 to n-1 do
      a[i] := (a[i] - mu)/sigma;
  //mu := mean_array(a,n);
  //sigma = sqrt(variance_array(a,n));
  {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.finish(opNorm);{$endif}
end;

function max_index(const a: PSingle; const n: longint): longint;
var i:longint; _max: single;
begin
    if n <= 0  then
      exit(-1);
    result := 0;
    _max  := a[0];
    for i := 1 to n-1 do
        if a[i] > _max then begin
            _max := a[i];
            result := i;
        end
end;

function min_index(const a: PSingle; const n: longint): longint;
var i:longint; _min: single;
begin
    if n <= 0  then
      exit(-1);
    result := 0;
    _min  := a[0];
    for i := 1 to n-1 do
        if a[i] < _min then begin
            _min := a[i];
            result := i;
        end
end;

function max_index(const a: TArray<Single>): longint;
begin
  //if n<0 then n:=Length(a);
  result:=max_index(PSingle(a),length(a))
end;

function min_index(const a: TArray<Single>): longint;
begin
  //if n<0 then n:=Length(a);
  result:=min_index(PSingle(a),length(a))
end;

function rand_normal():single;
const
  haveSpare : boolean = false;
  rand1     : double=0;
  rand2     : double=0;
begin

    if haveSpare then
    begin
        haveSpare := false;
        exit(sqrt(rand1) * sin(rand2))
    end;

    haveSpare := true;

    rand1 := trandom();// / ((double) RAND_MAX);
    if rand1 < 1e-100 then rand1 := 1e-100;
    rand1 := -2 * ln(rand1);
    rand2 := trandom() * TWO_PI;

    result := sqrt(rand1) * cos(rand2);
end;

function rand_uniform(const amin, amax: single):single;
begin
    if amax < amin then begin
      exit (trandom() * (amin - amax) + amax)
    end;
    result :=trandom() * (amax - amin) + amin
end;

procedure top_k(const a: TSingles; const n, k: longint; const index: PLongint);
var
    i, j, curr, swap: longint;
begin
    for j := 0 to k -1 do
        index[j] := -1;
    for i := 0 to n -1 do
        begin
            curr := i;
            for j := 0 to k -1 do
                if (index[j] < 0) or (a[curr] > a[index[j]]) then
                    begin
                        swap := curr;
                        curr := index[j];
                        index[j] := swap
                    end
        end
end;

function one_hot_encode(const a: TSingles; const n, k: longint): TSingles2d;
var
  i, index:longint;
begin
  //setLength(result,n);
  result:=AllocMem(n * sizeof(TSingles));
  for i := 0 to n-1 do begin
      result[i] := TSingles.Create(k);
      index := trunc(a[i]); // todo could round check C++ behavior?
      result[i][index] := 1;
  end;
end;

procedure one_hot_encode(const a: TArray<single>; const n, k: longint;
  var result: TArray<TArray<Single>>);
var
  i, index:longint;
begin
  setLength(result,n);
  //result:=AllocMem(n * sizeof(TSingles));
  for i := 0 to n-1 do begin
      setLength(result[i],k);
      //result[i] := TSingles.Create(k);
      index := trunc(a[i]); // todo could round check C++ behavior?
      result[i][index] := 1;
  end;
end;


function variance_array(const a:psingle; const n:longint):single;
var i:longint;
    sum, mean, v:single;
begin
  sum := 0;
  mean := mean_array(a, n);
    for i := 0 to n-1 do begin
        v  := a[i] - mean;
        sum := sum + v * v;
    end;
  result:= sum/n;
end;

function mse_array(const a: psingle; const n: integer): single;
var i:longint; sum:single;
begin
  sum := 0;
  for i := 0 to n-1 do
      sum := sum + a[i]*a[i];
  result := sqrt(sum/n);
end;

procedure transpose_matrix(const a:TSingles; const rows, cols: integer);
var
  transpose:TArray<single>;//TSingles;
  x, y:integer;
begin
  if rows*cols=0 then exit;
  setLength(transpose, rows*cols);//:=TSingles.Create(rows*cols);
  //setLength(transpose, rows*cols);
  for x := 0 to rows-1 do
      for y := 0to cols-1 do
          transpose[y*rows + x] := a[x*cols + y];
  move(transpose[0],a[0], rows*cols * sizeof(single));
  //transpose.free
end;

{$if defined(CPUX64) and defined(FPUAVX2)}
procedure mov(const N:PtrInt; const src, dst);assembler;{$ifdef FPC}nostackframe;local;{$endif}
asm
  {$ifndef FPC}
  .NOFRAME
  {$endif}
  mov     r11     ,  N
  shr     r11     ,  3
  jz      @rem
@while:
  vmovups ymm1     ,  [src]
  vmovups [dst]    ,  ymm1
  add     src      ,  32
  add     dst      ,  32
  dec     r11d
  jnz     @while
@rem:
  and     N ,  7
  jz      @done
@while2:
  vmovss  xmm1      ,  [src]
  vmovss  [dst]     ,  xmm1
  add     src       ,  4
  add     dst       ,  4
  dec     N
  jnz     @while2
@done:
end;
{$endif}




  procedure mov1(const f,t:PtrInt; const p:pointer=nil);
  var pa :PMpParams absolute p;
      src, dst:PSingle;
  begin
      src:=pa.A;
      dst:=pa.B;
      {$if defined(CPUX64) and defined(AVX2)}
      mov(t-f+1, src[f] , dst[f])
      {$else}
      move(src[f] , dst[f], (t-f+1) * sizeof(single))
      {$endif}
  end;

  procedure mov2(const f,t:PtrInt; const p:pointer=nil);
  var
    pa :PMpParams absolute p;
    i:PtrInt;
    INCX, INCY :Plongint;
    src, dst:PSingle;
  begin
    src := pa.A;
    dst := pa.B;
    INCX :=pa.C;
    INCY :=pa.D;
    for i := f to t do
        dst[i*INCY^] := src[i*INCX^];
  end;

procedure copy_cpu(const N: longint; const src: PSingle; const INCX: longint; const dst: PSingle; const INCY: longint);

var
  i: Integer;
  pars:TMPParams;
begin
  {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.start(opCopy);{$endif}
  pars.A:=src;
  pars.B:=dst;
  pars.C:=@INCX;
  pars.D:=@INCY;
  if (INCX=1) and (INCY=1) then
  {$if defined(USE_MULTITHREADING)}
      mp.&for(mov1,0,N-1,@pars)
  {$else}
      mov1(0, N-1, @pars)
  {$endif}
  else
  {$if defined(USE_MULTITHREADING)}
      mp.&for(mov2,0,N-1,@pars)
  {$else}
      mov2(0, N-1, @pars)
  {$endif}
  ;
  {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.finish(opCopy);{$endif}

end;

//procedure copy_cpu(const N: longint; const X: TSingles; const INCX: longint;
//  const Y: TSingles; const INCY: longint);
//var
//  i: Integer;
//begin
//
//  if (INCX=1) and (INCY=1) then begin
//    move(x[0],y[0],n*sizeof(single));
//    exit
//  end;
//
//  for i := 0 to n-1 do
//      Y[i*INCY] := X[i*INCX];
//end;

procedure reorg_cpu(const x: PSingle; const out_w, out_h, out_c, batch, stride: longint; const forward: boolean; const &out: PSingle);
var
  b,i,j,k:longint;
  in_c:longint;
  in_index, c2, offset, w2, h2, out_index:longint;
begin
  in_c := out_c div (stride*stride);

  for b := 0 to batch - 1 do
      for k := 0 to out_c - 1 do
          for j := 0 to out_h - 1 do
              for i := 0 to out_w - 1 do begin
                  in_index  := i + out_w*(j + out_h*(k + out_c*b));
                  c2 := k mod in_c;
                  offset := k div in_c;
                  w2 := i*stride + offset mod stride;
                  h2 := j*stride + offset div stride;
                  out_index := w2 + out_w*stride*(h2 + out_h*stride*(c2 + in_c*b));
                  if forward then
                    &out[out_index] := x[in_index]
                  else
                    &out[in_index] := x[out_index];
              end
end;

procedure scal_cpu(const N: longint; const ALPHA: single; const X: PSingle;
  const INCX: longint);
var i:longint;
  o:PSingle;
begin
  {$if defined(CPUX64) and defined(FPUAVX2)}
  if INCX=1 then begin
      smulvs(X, X, ALPHA, N);
      exit
  end;
  {$endif}

  for i := 0 to N-1 do begin
      o  := @X[i*INCX];
      o^ := o^ * ALPHA
  end;
end;

{$ifndef fpc}
procedure filldword(var X; const N:PtrInt; const a:longword);
var i:PtrInt;
    P:PLongword;
begin
  p:=@x;
  for I := 0 to N-1 do
    P[i]:=a
end;
{$endif}

{$if defined(CPUX64)and defined(FPUAVX2)}
procedure sfill( const x:PSingle; const N: PtrInt; const a:single);assembler;{$ifdef fpc}nostackframe;{$endif}
asm
  {$ifndef FPC}
  .NOFRAME
  {$endif}
  vbroadcastss ymm2, a

  mov     r11     ,  N
  shr     r11     ,  3
  jz      @rem
@while:
  vmovups [x]      ,  ymm2
  add     x        ,  32
  dec     r11d
  jnz     @while
@rem:
  and     N         ,  7
  jz      @done
@while2:
  vmovss  dword [x]       ,  xmm2
  add     x         ,  4
  dec     N
  jnz     @while2
@done:
end;
{$else}
procedure sfill( const x:PSingle; const N: PtrInt; const a:single);
var i:PtrInt;
begin
  for i:=0 to N-1 do
      x[i]:=a
end;
{$endif}

  procedure fill1(const f,t:PtrInt; const p:pointer=nil);inline;
  var a:PMPParams absolute p;
     ALPHA :PSingle;
     X:PSingle;
  begin
    ALPHA := a.A;
    X:=a.B;
    //sFill( x+f ,t-f+1,ALPHA^);
    FillDWord(x[f], t-f+1, PLongWord(ALPHA)^)
  end;

  procedure fill2(const f,t:PtrInt; const p:pointer=nil);inline;
  var
     i:integer;
     a:PMPParams absolute p;
     ALPHA :PSingle;
     INCX:PLongint;
     X:PSingle;
  begin
    ALPHA := a.A;
    X:=a.B;
    INCX:=a.C;
    for i := f to t do
      X[i*INCX^] := ALPHA^;
  end;

procedure fill_cpu(const N: longint; const ALPHA: single; const X: PSingle; const INCX: longint);
var p:TMPParams;
begin
  p.A:=@ALPHA;
  p.B:=X;
  P.C:=@INCX;

  if incx=1 then begin
    {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.start(opFill);{$endif}
    {$if defined(USE_MULTITHREADING)}
    mp.&For(fill1,0,N-1,@p);
    {$else}
    //fill1(0, N-1, @p);
    FillDWord(X[0],N,PLongword(@Alpha)^);
    {$endif}
    {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.finish(opFill);{$endif}
  end
  else begin
    {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.start(opIncFill);{$endif}
    //fill2(0,N-1)
    {$if defined(USE_MULTITHREADING)}
    mp.&For(fill2,0,N-1,@p);
    {$else}
    fill2(0, N-1, @p);
    {$endif}
    {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.finish(opIncFill);{$endif}
  end;
end;

// todo continue blas
procedure flatten(const x: PSingle; const size, layers, batch: longint; const forward: boolean);
var
  swap: TArray<single>;//TSingles;
  i, c, b, i1, i2: longint;
begin
  setLength(swap, size*layers*batch);
  //swap:=TSingles.Create(size*layers*batch);
  for b := 0 to  batch-1 do
      for c := 0 to layers-1do
          for i := 0 to size-1 do begin
              i1 := b*layers*size + c*size + i;
              i2 := b*layers*size + i*layers + c;
              if (forward) then
                swap[i2] := x[i1]
              else
                swap[i1] := x[i2];
           end;
  move(swap[0], x[0], size * layers * batch *sizeof(single));
  //swap.free
end;

function random_matrix(const rows, cols: longint): TSingles;
var i:longint;
begin
  //setLength(result,rows*cols);
  result:=TSingles.Create(rows*cols);
  for i := 0 to result.high() do
      result[i] := tRandom();

end;

procedure axpy_cpu(const N: longint; const ALPHA: single; const X: PSingle;
  const INCX: longint; const Y: PSingle; const INCY: longint);
var i:longint;
  o:PSingle;
begin
  {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.start(opAxpy);{$endif}
  // todo SIMDfy
  for i := 0 to N-1 do begin
      o := @Y[i*INCY];
      o^ := o^ + ALPHA*X[i*INCX];
  end;
  {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.start(opAxpy);{$endif}
end;

function dot_cpu(const N: longint; const X: PSIngle; const INCX: longint;
  const Y: PSingle; const INCY: longint):single;
var i:longint;
begin
  // todo SIMDfy
  result := 0;
  for i := 0 to N-1 do
      result := result + X[i*INCX] * Y[i*INCY];
end;

procedure test_blas;
begin
  // todo test_blas?
end;

procedure inter_cpu(const NX: longint; const X: PSingle; const NY: longint;
  const Y: PSingle; const B: longint; const &out: PSingle);
var i, j, index:longint;
begin
    index := 0;
    for j := 0 to  B-1 do begin
        for i := 0 to NX-1 do begin
            &OUT[index] := X[j*NX + i];inc(index);
        end;
        for i := 0 to NY-1 do begin
            &OUT[index] := Y[j*NY + i];inc(index)
        end
    end
end;

procedure deinter_cpu(const NX: longint; const X: PSingle; const NY: longint;
  const Y: PSingle; const B: longint; const &out: PSingle);
var  i, j, index:longint;
begin
    index := 0;
    for j := 0 to B -1 do begin
        for i := 0 to NX-1 do begin
            if assigned(X) then X[j*NX + i] := X[j*NX + i] + &out[index];
            inc(index);
        end;
        for i := 0 to NY-1 do begin
            if assigned(Y) then Y[j*NY + i] := Y[j*NY + i] + &out[index];
            inc(index)
        end
    end
end;

// note FMA
procedure mult_add_into_cpu(const N: longint; const X, Y, Z: PSingle);
var i:longint;
begin
  //todo SIMDfy
  for i := 0 to N-1 do
      Z[i] := Z[i] + X[i]*Y[i];
end;

// note Fill
procedure const_cpu(const N: longint; const ALPHA: single; const X: PSingle;
  const INCX: longint);
var
  i: Integer;
begin
    if N=0 then exit;
    if INCX=1 then begin
      fillDWord(x[0], N, PlongWord(@ALPHA)^);
      exit
    end;

    for i := 0 to  N-1 do
      X[i*INCX] := ALPHA;
end;

procedure pow_cpu(const N: longint; const ALPHA: single; const X: PSingle;
  const INCX: longint; const Y: PSingle; const INCY: longint);
var
  i: Integer;
begin
    for i := 0 to N-1 do
      Y[i*INCY] := Power(X[i*INCX], ALPHA)
end;

procedure mul_cpu(const N: longint; const X: PSingle; const INCX: longint;
  const Y: PSingle; const INCY: longint);
var
  i: Integer;
  o:PSingle;
begin
    // todo SIMDfy
  for i := 0 to N-1 do begin
    o  := @Y[i*INCY];
    o^ := o^ * X[i*INCX]
  end;
end;

function sum_array(const a:PSingle; const n:longint):single;
var i:longint;
begin
  // todo SIMDFy
    result := 0;
    for i := 0 to n-1 do
        result := result+ a[i];
end;

function mean_array(const a:PSingle; const n:longint):single;
begin
    result:=sum_array(a,n)/n
end;

procedure backward_bias(const bias_updates: PSingle; const delta: PSingle;
  const batch: longint; const n: longint; const size: longint);
var
    i, b: longint;
begin
    // todo simdfy
    for b := 0 to batch -1 do
        for i := 0 to n -1 do
            bias_updates[i] := bias_updates[i] + sum_array(delta+size * (i+b * n), size)
end;

procedure shortcut_cpu(const batch, w1, h1, c1: longint; const add: PSingle;
  const w2, h2, c2: longint; const &out: PSingle);
var
    stride: integer;
    sample: integer;
    minw: integer;
    minh: integer;
    minc: integer;
    i: integer;
    j: integer;
    k: integer;
    b: integer;
    out_index: integer;
    add_index: integer;
begin
    stride := w1 div w2;
    sample := w2 div w1;
    assert(stride = h1 div h2);
    assert(sample = h2 div h1);
    if stride < 1 then
        stride := 1;
    if sample < 1 then
        sample := 1;
    if (w1 < w2) then
        minw := w1
    else
        minw := w2;
    if (h1 < h2) then
        minh := h1
    else
        minh := h2;
    if (c1 < c2) then
        minc := c1
    else
        minc := c2;
    for b := 0 to batch -1 do
        for k := 0 to minc -1 do
            for j := 0 to minh -1 do
                for i := 0 to minw -1 do
                    begin
                        out_index := i * sample+w2 * (j * sample+h2 * (k+c2 * b));
                        add_index := i * stride+w1 * (j * stride+h1 * (k+c1 * b));
                        &out[out_index] := &out[out_index] + add[add_index];
                        //&out[out_index] := s1 * &out[out_index]+s2 * add[add_index]
                    end
end;

procedure mean_cpu(const x: PSingle; const batch, filters, spatial: longint;
  const mean: PSingle);
var
    scale : single;
    i, j, k, index:longint;
begin
    scale := 1/(batch * spatial);
    for i := 0 to filters - 1 do begin
        mean[i] := 0;
        for j := 0 to batch - 1 do
            for k := 0 to spatial - 1 do begin
                index := j*filters*spatial + i*spatial + k;
                mean[i] := mean[i] + x[index];
            end;
        mean[i] := mean[i]*scale;
    end

end;

procedure variance_cpu(const x, mean: PSingle; const batch, filters,
  spatial: longint; const variance: PSingle);
var
    scale : single;
    i, j, k, index :longint;
begin
    scale := 1/(batch * spatial - 1);
    for i := 0 to filters-1 do begin
        variance[i] := 0;
        for j := 0 to batch-1 do
            for k := 0 to spatial-1 do begin
                index := j*filters*spatial + i*spatial + k;
                variance[i] := variance[i] + sqr(x[index] - mean[i])
            end;

        variance[i] :=variance[i] * scale;
    end
end;


procedure backward_scale_cpu(const x_norm, delta: PSingle; const batch, n,
  size: longint; const scale_updates: PSingle);
var  i,b,f, index:longint;
    sum:single;
begin
  for f := 0 to n-1 do begin
      sum := 0;
      for b := 0 to batch-1 do
          for i := 0 to size-1 do begin
              index := i + size*(f + n*b);
              sum := sum + delta[index] * x_norm[index];
          end;
      scale_updates[f] := scale_updates[f] + sum;
  end
end;

procedure mean_delta_cpu(const delta, variance: PSingle; const batch, filters,
  spatial: longint; const mean_delta: PSingle);
var  i,j,k, index: longint;
begin
  for i := 0 to filters-1 do begin
      mean_delta[i] := 0;
      for j := 0 to batch-1 do
          for k := 0 to spatial-1 do begin
              index := j*filters*spatial + i*spatial + k;
              mean_delta[i] := mean_delta[i]+delta[index];
          end;
      mean_delta[i] :=  mean_delta[i] * (-1./sqrt(variance[i] + 0.00001 ));
  end
end;

procedure variance_delta_cpu(const x, delta, mean, variance: PSingle;
  const batch, filters, spatial: longint; const variance_delta: PSingle);
var i,j,k, index:longint;
begin
  for i := 0 to filters-1 do begin
      variance_delta[i] := 0;
      for j := 0 to batch-1 do
          for k := 0 to spatial-1 do begin
              index := j*filters*spatial + i*spatial + k;
              variance_delta[i] := variance_delta[i]+delta[index]*(x[index] - mean[i]);
          end;
      variance_delta[i] := variance_delta[i] * -0.5 * Power(variance[i] + 0.00001 , -3.0/2.0);
  end

end;

procedure normalize_delta_cpu(const x, mean, variance, mean_delta,
  variance_delta: PSingle; const batch, filters, spatial: longint;
  const delta: PSingle);
var  f, j, k, index:longint;
begin
  for j := 0 to  batch-1 do
      for f := 0 to filters-1 do
          for k := 0 to spatial-1 do begin
              index := j*filters*spatial + f*spatial + k;
              delta[index] := delta[index] * 1/(sqrt(variance[f] + 0.00001 )) + 2 * variance_delta[f] * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
          end;
end;

{$if defined(CPUX64) and defined(AVX2)}
procedure normalize(const x:PSingle; const mean, variance:single ;const N:longint);assembler;{$ifdef fpc}nostackframe;{$endif}
const EPS : single = 0.000001;//_EPSILON;
asm
  //vzeroupper
    //vaddss       variance  , variance  , [rip + EPS]     //
    rsqrtss      variance  , variance      // variance = 1/sqrt(variance)
    vbroadcastss ymm2      , variance
    vbroadcastss ymm1      , mean

    mov          r11d      , N
    shr          r11d      , 3
    jz           @rem
@while:
    vmovups      ymm3      , [x]
    vsubps       ymm3      , ymm3      , ymm1
    vmulps       ymm3      , ymm3      , ymm2
    vmovups      [x]       , ymm3
    add          x         , 32
    dec          r11d
    jnz          @while

@rem:
    and          N         , 7
    jz           @done
@while2:
    vmovss       xmm3      , [x]
    vsubss       xmm3      , xmm3      , xmm1
    vmulss       xmm3      , xmm3      , xmm2
    vmovss       [x]       , xmm3
    add          x         , 4
    dec          N
    jnz          @while2

@done:

end;
{$endif}


procedure normalize_cpu(const x, mean, variance: PSingle; const batch, filters, spatial: longint);
var b, j, index: longint;
    i  :longint;
    o : PSingle;
begin
    for b := 0 to batch-1 do
        for j := 0 to filters-1 do begin
        {$if defined(CPUX64) and defined(AVX2)}
            index := b*filters*spatial + j*spatial;
            normalize(@x[index], mean[j], variance[j], spatial)
        {$else}
            for i := 0 to spatial-1 do begin
                o := @x[b*filters*spatial + j*spatial + i];
                o^ := (o^ - mean[j])/(sqrt(variance[j]) + 0.000001)
            end
        {$endif}
        end;
end;

procedure scale_bias(const output, scales: PSingle; const batch, n, size: longint);
var
    i, j, b : longint;
    o : PSingle;
begin
  {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.start(opBatchMulvs);{$endif}
  for b := 0 to batch-1 do
      for i := 0 to n-1 do begin
          o:=@output[(b*n + i)*size];
      {$if defined(CPUX64) and defined(AVX2)}
          smulvs(o, o, scales[i], size);
      {$else}
          for j := 0 to size -1 do begin
              o[j] := o[j] * scales[i];
          end;
      {$endif}
      end;
  {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.finish(opBatchMulvs);{$endif}
end;

procedure add_bias(const output: PSingle; const biases: PSingle; const batch: longint; const n: longint; const size: longint);
var
    i, j, b: longint;
    o:PSingle;
begin
    {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.start(opBatchAddvs);{$endif}
    for b := 0 to batch -1 do
        for i := 0 to n -1 do begin
            o := @output[(b * n+i) * size];
        {$if defined(CPUX64) and defined(FPUAVX2)}
            saddvs(o, o, biases[i], size);
        {$else}
            for j := 0 to size -1 do begin
                o[j] := o[j] + biases[i]
            end;
        {$endif}
        end;

    {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.finish(opBatchAddvs);{$endif}
end;

{$if defined(CPUX64) and defined(AVX2)}
procedure sfmaddvss(const x:PSingle ; const s, b:single; N:longint);assembler;{$ifdef def}nostackframe;{$endif}
asm
  vbroadcastss ymm0      , s
  vbroadcastss ymm1      , b

  mov          r11d       , N
  shr          r11       , 3             // div regs
  jz           @rem

@while1:

  vmovups      ymm2      , [x]
  vfmadd213ps  ymm2      , ymm0   ,  ymm1       // x := x * s + b
  vmovups      [x]       , ymm2

  add          x         , 32
  dec          r11
  jnz          @while1

  @rem:
  //mov          r11       , N
  and          N         , 7       // mod regs
  jz           @done

@while2:
  vmovss       xmm2      , [x]
  vfmadd213ss  xmm2      , xmm0   ,  xmm1
  vmovss       [x]       , xmm2
  add          x         , 4

  dec          N
  jnz          @while2

@done:
end;
{$endif}

procedure scale_add_bias(const output, scales, biases: PSingle; const batch, n, size: longint);
var
    b, i, j : longint;
    o : PSingle;
begin
  {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.start(opBatchFma);{$endif}
  for b := 0 to batch-1 do
      for i := 0 to n-1 do begin
      {$if defined(CPUX64) and defined(AVX2)}
          sfmaddvss(@output[(b * n + i) * size], scales[i], biases[i], size);
      {$else}
          for j := 0 to size -1 do begin
              o:=@output[(b * n+i)*size + j];
              o^ := o^ * scales[i];
              o^ := o^ + biases[i]
          end;
      {$endif}
      end;
  {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.finish(opBatchFma);{$endif}
end;

procedure scal_add_cpu(const N:longint; const ALPHA, BETA:single; const X:PSingle;const INCX:longint);
var i:longint;
begin
  {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.start(opFma);{$endif}

  {$if defined(CPUX64) and defined(AVX2)}
  if INCX=1 then
      begin
          sfmaddvss(x, ALPHA, BETA, N);
          exit();
      end;
  {$endif}
  for i := 0 to N -1 do
    X[i*INCX] := X[i*INCX] * ALPHA + BETA;
  {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.finish(opFma);{$endif}
end;


procedure l2normalize_cpu(const x, dx: PSingle; const batch, filters,
  spatial: longint);
var  b,f,i, index:longint;
    sum:single;
begin
  // todo simdfy
  for b := 0 to batch-1 do
      for i := 0 to spatial-1 do begin
          sum := 0;
          for f := 0 to filters-1 do begin
              index := b*filters*spatial + f*spatial + i;
              sum := sum + sqr(x[index]{, 2});
          end;
          sum := sqrt(sum);
          for f := 0 to filters-1 do begin
              index := b*filters*spatial + f*spatial + i;
              x[index] := x[index] / sum;
              dx[index] := (1 - x[index]) / sum;
          end
      end;
end;

procedure smooth_l1_cpu(const n: longint; const pred, truth, delta, error: PSingle);
var i:longint;
    diff,abs_val:single;
begin
  //todo simdfy
  for i := 0 to n-1 do begin
      diff := truth[i] - pred[i];
      abs_val := abs(diff);
      if abs_val < 1 then begin
          error[i] := diff * diff;
          delta[i] := diff;
      end
      else begin
          error[i] := 2*abs_val - 1;
          if diff < 0 then
              delta[i] := 1
          else
              delta[i] := -1
      end;
  end;

end;

procedure l2_cpu(const n: longint; const pred, truth, delta, error: PSingle);
var i:longint;
    diff:single;
begin
  // todo simdfy
  for i := 0 to n-1 do begin
      diff := truth[i] - pred[i];
      error[i] := diff * diff;
      delta[i] := diff
  end
end;

procedure l1_cpu(const n: longint; const pred, truth, delta, error: PSingle);
var i:longint;
    diff:single;
begin
  // todo simdfy
  for i := 0 to n-1 do begin
      diff := truth[i] - pred[i];
      error[i] := abs(diff);
      if diff > 0 then
          delta[i] := 1
      else
          delta[i]:= -1
  end
end;

procedure logistic_x_ent_cpu(const n: longint; const pred, truth, delta, error: PSingle);
var i:longint;
    t,p:single;
begin
  // todo simdfy
  for i := 0 to n-1 do begin
      t := truth[i];
      p := pred[i];
      error[i] := -t*ln(p) - (1-t)*ln(1-p);
      delta[i] := t-p;
  end
end;

procedure softmax_x_ent_cpu(const n: longint; const pred, truth, delta, error: PSingle);
var i:longint;
    t,p :single;
begin
  //todo simdfy
  for i := 0 to n-1do begin
      t := truth[i];
      p := pred[i];
      if t<>0 then
          error[i] := -ln(p)
      else
          error[i] := 0;
      delta[i] := t-p;
  end
end;

function ifthen( const when: boolean; const this,that:single):single;inline;overload;
begin
  if when then
      result:=this
  else
      result:=that
end;

procedure weighted_sum_cpu(const a, b, s: PSingle; const n: longint;
  const c: PSingle);
var  i:longint;
begin
  // todo simdfy
  if assigned(b) then
    for  i := 0 to n-1 do
      c[i] := s[i]*a[i] + (1-s[i]) * b[i]
  else
    for  i := 0 to n-1 do
      c[i] := s[i]*a[i]

end;

procedure weighted_delta_cpu(const a, b, s, da, db, ds: PSingle;
  const n: longint; const dc: PSingle);
var i:longint;
begin
  // too simdfy
  for i := 0 to n-1 do begin
      if assigned(da) then da[i] := da[i] + dc[i] * s[i];
      if assigned(db) then db[i] := db[i] + dc[i] * (1-s[i]);
      ds[i] := ds[i] + dc[i] * (a[i] - b[i]);
  end
end;

function relu(const x:single):single;
begin
  result := x*longint(x>0)
end;

type
  PSTParams = ^TSTParams;
  TSTParams = record
     src_outputs:longint;
     weights_normalization:TWeightsNormalization;
     weights, &in, &out:PSingle;
     N, layer_step, step : longint;
     outputs_of_layers: TArray<longint>;
     layers_output:PPsingle;

  end;

procedure shortThread(const f,t:PtrInt; const p:pointer);
var
    id, src_id, src_i, src_b, i, weights_index, add_outputs, add_index, out_index: longint;
    sum, max_val, w, eps: single; add:PSingle;
    a:PSTParams absolute p;
begin
    for id := f to t do
        begin
            src_id := id;
            src_i := src_id mod a.src_outputs;
            src_id := src_id div a.src_outputs;
            src_b := src_id;
            sum := 1; max_val := -MaxSingle;
            if assigned(a.weights) and boolean(a.weights_normalization) then
                begin
                    if a.weights_normalization = wnSOFTMAX_NORMALIZATION then
                        for i := 0 to (a.n+1) -1 do
                            begin
                                weights_index := src_i div a.step+i * a.layer_step;
                                w := a.weights[weights_index];
                                if max_val < w then
                                    max_val := w
                            end;
                    eps := 0.0001;
                    sum := eps;
                    for i := 0 to (a.n+1) -1 do
                        begin
                            weights_index := src_i div a.step+i * a.layer_step;
                            w := a.weights[weights_index];
                            if a.weights_normalization = wnRELU_NORMALIZATION then
                                sum := sum + relu(w)
                            else
                                if a.weights_normalization = wnSOFTMAX_NORMALIZATION then
                                    sum := sum + exp(w-max_val)
                        end
                end;
            if assigned(a.weights) then
                begin
                    w := a.weights[src_i div a.step];
                    if a.weights_normalization = wnRELU_NORMALIZATION then
                        w := relu(w) / sum
                    else
                        if a.weights_normalization = wnSOFTMAX_NORMALIZATION then
                            w := exp(w-max_val) / sum;
                    a.&out[id] := a.&in[id] * w
                end
            else
                a.&out[id] := a.&in[id];
            for i := 0 to a.n -1 do
                begin
                    add_outputs := a.outputs_of_layers[i];
                    if src_i < add_outputs then
                        begin
                            add_index := add_outputs * src_b+src_i;
                            out_index := id;
                            add := a.layers_output[i];
                            if assigned(a.weights) then
                                begin
                                    weights_index := src_i div a.step+(i+1) * a.layer_step;
                                    w := a.weights[weights_index];
                                    if a.weights_normalization = wnRELU_NORMALIZATION then
                                        w := relu(w) / sum
                                    else
                                        if a.weights_normalization = wnSOFTMAX_NORMALIZATION then
                                            w := exp(w-max_val) / sum;
                                    a.&out[out_index] := a.&out[out_index] + (add[add_index] * w)
                                end
                            else
                                a.&out[out_index] := a.&out[out_index] + add[add_index]
                        end
                end
        end;
end;


procedure shortcut_multilayer_cpu(size: longint; src_outputs: longint;
  batch: longint; n: longint; outputs_of_layers: TArray<longint>;
  layers_output: PPsingle; &out: Psingle; &in: Psingle; weights: Psingle;
  nweights: longint; weights_normalization: TWeightsNormalization);
var
    a:TSTParams;

begin
    a.src_outputs:=src_outputs;
    a.weights_normalization:=weights_normalization;
    a.weights:=weights;
    a.&in:=&in;
    a.&out:=&out;
    a.N:=N;
    a.step:=0;
    a.outputs_of_layers:=outputs_of_layers;
    a.layers_output:=layers_output;
    a.layer_step := nweights div (n+1);
    if (nweights > 0) then
        a.step := src_outputs div a.layer_step;
    {$if defined(USE_MULTITHREADING)}
    mp.&for(shortThread, 0, size-1, @a)   ;
    {$else}
    shortThread(0, size -1, @a)
    {$endif}
    // todo SIMDfy
end;

procedure backward_shortcut_multilayer_cpu(size: longint; src_outputs: longint;
  batch: longint; n: longint; outputs_of_layers: TArray<longint>;
  layers_delta: PPsingle; delta_out: Psingle; delta_in: Psingle;
  weights: Psingle; weight_updates: Psingle; nweights: longint; &in: Psingle;
  layers_output: PPsingle; weights_normalization: TWeightsNormalization);
var
    layer_step: longint;
    step: longint;
    id: longint;
    src_id: longint;
    src_i: longint;
    src_b: longint;
    grad, sum, max_val, w, eps: single;
    i: longint;
    weights_index: longint;
    add, layer_delta : PSingle;
    add_outputs: longint;
    add_index: longint;
    out_index: longint;
begin
    layer_step := nweights div (n+1);
    step := 0;
    if (nweights > 0) then
        step := src_outputs div layer_step;
    // todo SIMDfy
    for id := 0 to size -1 do
        begin
            src_id := id;
            src_i := src_id mod src_outputs;
            src_id := src_id div src_outputs;
            src_b := src_id;
            grad := 1; sum := 1; max_val := -MaxSingle;
            if assigned(weights) and boolean(weights_normalization) then
                begin
                    if weights_normalization = wnSOFTMAX_NORMALIZATION then
                        for i := 0 to (n+1) -1 do
                            begin
                                weights_index := src_i div step+i * layer_step;
                                w := weights[weights_index];
                                if max_val < w then
                                    max_val := w
                            end;
                    eps := 0.0001;
                    sum := eps;
                    for i := 0 to (n+1) -1 do
                        begin
                            weights_index := src_i div step+i * layer_step;
                            w := weights[weights_index];
                            if weights_normalization = wnRELU_NORMALIZATION then
                                sum := sum + relu(w)
                            else
                                if weights_normalization = wnSOFTMAX_NORMALIZATION then
                                    sum := sum + exp(w-max_val)
                        end
                end;
            if assigned(weights) then
                begin
                    w := weights[src_i div step];
                    if weights_normalization = wnRELU_NORMALIZATION then
                        w := relu(w) / sum
                    else
                        if weights_normalization = wnSOFTMAX_NORMALIZATION then
                            w := exp(w-max_val) / sum;
                    delta_out[id] := delta_out[id] + (delta_in[id] * w);
                    weight_updates[src_i div step] := weight_updates[src_i div step] + (delta_in[id] * &in[id] * grad)
                end
            else
                delta_out[id] := delta_out[id] + delta_in[id];
            for i := 0 to n -1 do
                begin
                    add_outputs := outputs_of_layers[i];
                    if src_i < add_outputs then
                        begin
                            add_index := add_outputs * src_b+src_i;
                            out_index := id;
                            layer_delta := layers_delta[i];
                            if assigned(weights) then
                                begin
                                    add := layers_output[i];
                                    weights_index := src_i div step+(i+1) * layer_step;
                                    w := weights[weights_index];
                                    if weights_normalization = wnRELU_NORMALIZATION then
                                        w := relu(w) / sum
                                    else
                                        if weights_normalization = wnSOFTMAX_NORMALIZATION then
                                            w := exp(w-max_val) / sum;
                                    layer_delta[add_index] := layer_delta[add_index] + (delta_in[id] * w);
                                    weight_updates[weights_index] := weight_updates[weights_index] + (delta_in[id] * add[add_index] * grad)
                                end
                            else
                                layer_delta[add_index] := layer_delta[add_index] + delta_in[id]
                        end
                end
        end
end;

procedure softmax(const input: PSingle; const n: longint; const temp: single; const stride: longint; const output: PSingle);
var i:longint;
    sum, largest, e : single;
    o:PSingle;
begin
  sum := 0;
  largest := -MaxSingle;
  for i := 0 to n-1 do
      if input[i*stride] > largest then
          largest := input[i*stride];
  for i := 0 to n-1 do  begin
      e := exp(input[i*stride]/temp - largest/temp);
      sum := sum + e;
      output[i*stride] := e;
  end;
  for i := 0 to n-1 do  begin
      o:=@output[i*stride];
      o^ := o^ / sum;
  end;

end;

procedure softmax(const input: PSingle; const n: longint; const temp: single;
  const output: PSingle; const stride: longint);
begin
  softmax(input, n, temp, stride, output);
end;

procedure softmax_cpu(const input: PSingle; const n, batch, batch_offset,
  groups, group_offset, stride: longint; const temp: single;
  const output: PSingle);
var g, b:longint;
begin
  for b := 0 to batch-1 do
      for g := 0 to groups-1 do
          softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
end;

procedure upsample_cpu(const &in: PSingle; const w, h, c, batch,
  stride: longint; const forward: boolean; const scale: single;
  const &out: PSingle);
var i, j, k, b, in_index, out_index:longint;
begin
   for b := 0 to batch-1 do
       for k := 0 to c-1 do
           for j := 0 to h*stride-1 do
               for i := 0 to w*stride-1 do begin
                   in_index := b*w*h*c + k*w*h + (j div stride)*w + i div stride;
                   out_index := b*w*h*c*stride*stride + k*w*h*stride*stride + j*w*stride + i;
                   if forward then
                     &out[out_index] := scale*&in[in_index]
                   else
                     &in[in_index] := &in[in_index] + scale*&out[out_index]
               end
end;

procedure constrain_cpu(size: longint; ALPHA: single; X: Psingle);
var
    i: longint;
begin
    for i := 0 to size -1 do
        X[i] := math.min(ALPHA, math.max(-ALPHA, X[i]))
end;

procedure fix_nan_and_inf_cpu(input: Psingle; size: size_t);
var
    i: longint;
    val: single;
begin
    for i := 0 to size -1 do
        begin
            val := input[i];
            if IsNan(val) or IsInfinite(val) then
                input[i] := 1.0 / i
        end
end;

procedure get_embedding(src: Psingle; src_w: longint; src_h: longint; src_c: longint; embedding_size: longint; cur_w: longint; cur_h: longint; cur_n: longint; cur_b: longint; dst: Psingle);
var
    i: longint;
    src_index: longint;
    val: single;
begin
    for i := 0 to embedding_size -1 do
        begin
            src_index := cur_b * (src_c * src_h * src_w)+cur_n * (embedding_size * src_h * src_w)+i * src_h * src_w+cur_h * (src_w)+cur_w;
            val := src[src_index];
            dst[i] := val
        end
end;

function math_vector_length(A: Psingle; feature_size: Longword):single;
var
    sum: single;
    i: longint;
    vector_length: single;
begin
    sum := 0;
    for i := 0 to feature_size -1 do
        sum := sum + (A[i] * A[i]);
    vector_length := sqrt(sum);
    exit(vector_length)
end;

function cosine_similarity(A: Psingle; B: Psingle; feature_size: Longword):single;
var
    mul, d_a, d_b: single;
    i: longint;
    similarity: single;
    divider: single;
begin
    mul := 0.0; d_a := 0.0; d_b := 0.0;
    for i := 0 to feature_size -1 do
        begin
            mul := mul + (A[i] * B[i]);
            d_a := d_a + (A[i] * A[i]);
            d_b := d_b + (B[i] * B[i])
        end;
    divider := sqrt(d_a) * sqrt(d_b);
    if divider > 0 then
        similarity := mul / divider
    else
        similarity := 0;
    exit(similarity)
end;

function get_sim_P_index(i: size_t; j: size_t; contrast_p: PContrastiveParams;
  contrast_p_size: longint): longint;
var
    z: size_t;
begin
    for z := 0 to contrast_p_size -1 do
        if (contrast_p[z].i = i) and (contrast_p[z].j = j) then
            break;
    if z = contrast_p_size then
        exit(-1);
    exit(z)
end;

function check_sim(i: size_t; j: size_t; contrast_p: PContrastiveParams;
  contrast_p_size: longint): longint;
var
    z: size_t;
begin
    for z := 0 to contrast_p_size -1 do
        if (contrast_p[z].i = i) and (contrast_p[z].j = j) then
            break;
    if z = contrast_p_size then
        exit(0);
    exit(1)
end;

function find_sim(i: size_t; j: size_t; contrast_p: PContrastiveParams;
  contrast_p_size: longint): single;
var
    z: size_t;
begin
    for z := 0 to contrast_p_size -1 do
        if (contrast_p[z].i = i) and (contrast_p[z].j = j) then
            break;
    if z = contrast_p_size then
        begin
            writeln(format(' Error: find_sim(): sim isn''t found: i = %zu, j = %zu, z = %zu ',[ i, j, z]));
            raise exception.Create('Error!')
        end;
    exit(contrast_p[z].sim)
end;

function find_P_constrastive(i: size_t; j: size_t;
  contrast_p: PContrastiveParams; contrast_p_size: longint): single;
var
    z: size_t;
begin
    for z := 0 to contrast_p_size -1 do
        if (contrast_p[z].i = i) and (contrast_p[z].j = j) then
            break;
    if z = contrast_p_size then
        begin
            writeln(format(' Error: find_P_constrastive(): P isn''t found: i = %zu, j = %zu, z = %zu ', [i, j, z]));
            raise exception.Create('Error!')
        end;
    exit(contrast_p[z].P)
end;

function P_constrastive_f_det(il: size_t; labels: Plongint; z: PPsingle;
  feature_size: Longword; temperature: single; contrast_p: PContrastiveParams;
  contrast_p_size: longint): single;
var
    sim: single;
    i: size_t;
    j: size_t;
    numerator: single;
    denominator: single;
    k: longint;
    cp: TContrastiveParams;
    &_result: single;
begin
    sim := contrast_p[il].sim;
    i := contrast_p[il].i;
    j := contrast_p[il].j;
    numerator := exp(sim / temperature);
    denominator := 0;
    for k := 0 to contrast_p_size -1 do
        begin
            cp := contrast_p[k];
            if (cp.i <> i) and (cp.j = j) then
                denominator := denominator + cp.exp_sim
        end;
    &_result := 0.9999;
    if denominator <> 0 then
        &_result := numerator / denominator;
    if &_result > 1 then
        &_result := 0.9999;
    exit(&_result)
end;

function P_constrastive_f(i: size_t; l: size_t; labels: Plongint; z: PPsingle;
  feature_size: Longword; temperature: single; contrast_p: PContrastiveParams;
  contrast_p_size: longint): single;
var
    sim: single;
    numerator: single;
    denominator: single;
    k: longint;
    cp: TContrastiveParams;
    &_result: single;
begin
    if (i = l) then
        begin
            writeln(ErrOutput, format(' Error: in P_constrastive must be i != l, while i = %zu, l = %zu ', [i, l]));
            raise exception.Create('Error!')
        end;
    sim := find_sim(i, l, contrast_p, contrast_p_size);
    numerator := exp(sim / temperature);
    denominator := 0;
    for k := 0 to contrast_p_size -1 do
        begin
            cp := contrast_p[k];
            if (cp.i <> i) and (cp.j = l) then
                denominator := denominator + cp.exp_sim
        end;
    &_result := 0.9999;
    if denominator <> 0 then
        &_result := numerator / denominator;
    if &_result > 1 then
        &_result := 0.9999;
    exit(&_result)
end;

procedure grad_contrastive_loss_positive_f(i: size_t; class_ids: Plongint;
  labels: Plongint; num_of_samples: size_t; z: PPsingle;
  feature_size: Longword; temperature: single; delta: Psingle; wh: longint;
  contrast_p: PContrastiveParams; contrast_p_size: longint);
var
    vec_len: single;
    j: size_t;
    N: size_t;
    mult: single;
    sim_P_i: longint;
    sim: single;
    P: single;
    m: longint;
    d: single;
    out_i: longint;
begin
    vec_len := math_vector_length(z[i], feature_size);
    N := 0;
    for j := 0 to num_of_samples -1 do
        if (labels[i] = labels[j]) and (labels[i] >= 0) then
            inc(N);
    if (N = 0) or (temperature = 0) or (vec_len = 0) then
        begin
            writeln(ErrOutput, format(' Error: N == 0 || temperature == 0 || vec_len == 0. N=%f, temperature=%f, vec_len=%f, labels[i] = %d ', [N, temperature, vec_len, labels[i]]));
            raise exception.Create('Error!')
        end;
    mult := 1 / ((N-1) * temperature * vec_len);
    for j := 0 to num_of_samples -1 do
        if (i <> j) and (labels[i] = labels[j]) and (labels[i] >= 0) then
            begin
                sim_P_i := get_sim_P_index(i, j, contrast_p, contrast_p_size);
                if sim_P_i < 0 then
                    continue;
                sim := contrast_p[sim_P_i].sim;
                P := contrast_p[sim_P_i].P;
                for m := 0 to feature_size -1 do
                    begin
                        d := mult * (sim * z[i][m]-z[j][m]) * (1-P);
                        out_i := m * wh;
                        delta[out_i] := delta[out_i] - d
                    end
            end
end;

procedure grad_contrastive_loss_negative_f(i: size_t; class_ids: Plongint;
  labels: Plongint; num_of_samples: size_t; z: PPsingle;
  feature_size: Longword; temperature: single; delta: Psingle; wh: longint;
  contrast_p: PContrastiveParams; contrast_p_size: longint; neg_max: longint);
var
    vec_len: single;
    j: size_t;
    N: size_t;
    mult: single;
    neg_counter: longint;
    k: size_t;
    sim_P_i: longint;
    sim: single;
    P: single;
    m: longint;
    d: single;
    out_i: longint;
begin
    vec_len := math_vector_length(z[i], feature_size);
    N := 0;
    for j := 0 to num_of_samples -1 do
        if (labels[i] = labels[j]) and (labels[i] >= 0) then
            inc(N);
    if (N = 0) or (temperature = 0) or (vec_len = 0) then
        begin
            writeln(ErrOutput, format(' Error: N == 0 || temperature == 0 || vec_len == 0. N=%f, temperature=%f, vec_len=%f, labels[i] = %d ', [N, temperature, vec_len, labels[i]]));
            raise exception.Create('Error!')
        end;
    mult := 1 / ((N-1) * temperature * vec_len);
    neg_counter := 0;
    for j := 0 to num_of_samples -1 do
        if (labels[i] >= 0) and (labels[i] = labels[j]) and (i <> j) then
            begin
                for k := 0 to num_of_samples -1 do
                    if (k <> i) and (k <> j) and (labels[k] <> labels[i]) and (class_ids[j] = class_ids[k]) then
                        begin
                            inc(neg_counter);
                            sim_P_i := get_sim_P_index(i, k, contrast_p, contrast_p_size);
                            if sim_P_i < 0 then
                                continue;
                            sim := contrast_p[sim_P_i].sim;
                            P := contrast_p[sim_P_i].P;
                            for m := 0 to feature_size -1 do
                                begin
                                    d := mult * (z[k][m]-sim * z[i][m]) * P;
                                    out_i := m * wh;
                                    delta[out_i] := delta[out_i] - d
                                end;
                            if neg_counter >= neg_max then
                                exit()
                        end
            end
end;

function P_constrastive(i: size_t; l: size_t; labels: Plongint; num_of_samples: size_t; z: PPsingle; feature_size: Longword; temperature: single; cos_sim: Psingle; exp_cos_sim: Psingle):single;
var
    numerator: single;
    denominator: single;
    k: longint;
    &_result: single;
begin
    if i = l then
        begin
            writeln(ErrOutput,format( ' Error: in P_constrastive must be i != l, while i = %zu, l = %zu ', [i, l]));
            raise exception.Create('Error!')
        end;
    numerator := exp_cos_sim[i * num_of_samples+l];
    denominator := 0;
    for k := 0 to num_of_samples -1 do
        if k <> i then
            denominator := denominator + exp_cos_sim[k * num_of_samples+l];
    &_result := numerator / denominator;
    exit(&_result)
end;

procedure grad_contrastive_loss_positive(i: size_t; labels: Plongint; num_of_samples: size_t; z: PPsingle; feature_size: Longword; temperature: single; cos_sim: Psingle; p_constrastive: Psingle; delta: Psingle; wh: longint);
var
    vec_len: single;
    j: size_t;
    N: size_t;
    mult: single;
    sim: single;
    P: single;
    m: longint;
    d: single;
    out_i: longint;
begin
    vec_len := math_vector_length(z[i], feature_size);
    N := 0;
    for j := 0 to num_of_samples -1 do
        if labels[i] = labels[j] then
            inc(N);
    if (N = 0) or (temperature = 0) or (vec_len = 0) then
        begin
            writeln(ErrOutput, format(' Error: N == 0 || temperature == 0 || vec_len == 0. N=%f, temperature=%f, vec_len=%f ', [N, temperature, vec_len]));
            raise exception.Create('Error!')
        end;
    mult := 1 / ((N-1) * temperature * vec_len);
    for j := 0 to num_of_samples -1 do
        if (i <> j) and (labels[i] = labels[j]) then
            begin
                sim := cos_sim[i * num_of_samples+j];
                P := p_constrastive[i * num_of_samples+j];
                for m := 0 to feature_size -1 do
                    begin
                        d := mult * (sim * z[i][m]-z[j][m]) * (1-P);
                        out_i := m * wh;
                        delta[out_i] := delta[out_i] - d
                    end
            end
end;

procedure grad_contrastive_loss_negative(i: size_t; labels: Plongint; num_of_samples: size_t; z: PPsingle; feature_size: Longword; temperature: single; cos_sim: Psingle; p_constrastive: Psingle; delta: Psingle; wh: longint);
var
    vec_len: single;
    j: size_t;
    N: size_t;
    mult: single;
    k: size_t;
    sim: single;
    P: single;
    m: longint;
    d: single;
    out_i: longint;
begin
    vec_len := math_vector_length(z[i], feature_size);
    N := 0;
    for j := 0 to num_of_samples -1 do
        if labels[i] = labels[j] then
            inc(N);
    if (N = 0) or (temperature = 0) or (vec_len = 0) then
        begin
            writeln(ErrOutput, format(' Error: N == 0 || temperature == 0 || vec_len == 0. N=%f, temperature=%f, vec_len=%f ', [N, temperature, vec_len]));
            raise exception.Create('Error!')
        end;
    mult := 1 / ((N-1) * temperature * vec_len);
    for j := 0 to num_of_samples -1 do
        if (i <> j) and (labels[i] = labels[j]) then
            begin
                for k := 0 to num_of_samples -1 do
                    if (k <> i) and (k <> j) and (labels[k] >= 0) then
                        begin
                            sim := cos_sim[i * num_of_samples+k];
                            P := p_constrastive[i * num_of_samples+k];
                            for m := 0 to feature_size -1 do
                                begin
                                    d := mult * (z[k][m]-sim * z[i][m]) * P;
                                    out_i := m * wh;
                                    delta[out_i] := delta[out_i] - d
                                end
                        end
            end
end;



{$ifdef GPU}
procedure constrain_gpu(N: longint; ALPHA: single; X: PSingle; INCX: longint);
begin

end;

function test_gpu_blas: longint;
begin

end;

{$endif}
initialization

  //normalize(@aa[0],1,2,length(aa));
  //writeln(TSingles(aa).toString(', ',length(aa)));
  //readln()

end.

