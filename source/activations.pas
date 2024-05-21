unit Activations;

{$ifdef fpc}
  {$mode delphi}
  {$ModeSwitch cvar}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
  {$ifdef CPUX64}
          {$FPUType AVX2}
          {$asmmode intel}
  {$endif}
{$endif}
{$pointermath on}

interface

uses
  Sysutils, Math, typinfo, lightnet;


function stair_activate(const x:single) :single;inline;
function hardtan_activate(const x:single):single;inline;
function linear_activate(const x:single):single;inline;
function logistic_activate(const x:single):single;//inline;
function loggy_activate(const x:single):single;inline;
function relu_activate(const x:single):single;inline;
function relu6_activate(const x:single):single;inline;
function elu_activate(const x:single):single;inline;
function selu_activate(const x:single):single;inline;
function gelu_activate(const x:single):single;inline;
function relie_activate(const x:single):single;inline;
function ramp_activate(const x:single):single;inline;
function tanh_activate(const x:single):single;inline;
function softplus_activate(const x, threshold : single):single;inline;
function plse_activate(const x:single):single;inline;
function lhtan_activate(const x:single):single;inline;
function lhtan_gradient(const x:single):single;inline;
function hardtan_gradient(const x:single):single;inline;
function linear_gradient(const x:single):single;inline;
function logistic_gradient(const x:single):single;inline;
function loggy_gradient(const x:single):single;inline;
function stair_gradient(const x:single):single;inline;
function relu_gradient(const x:single):single;inline;
function relu6_gradient(const x:single):single;inline;
function elu_gradient(const x:single):single;inline;
function selu_gradient(const x:single):single;inline;
function relie_gradient(const x:single):single;inline;
function ramp_gradient(const x:single):single;inline;
function leaky_gradient(const x:single):single;inline;
function tanh_gradient(const x:single):single;inline;
function sech(const x:single):single;inline;
function gelu_gradient(const x:single):single;inline;
function plse_gradient(const x:single):single;inline;


function leaky_activate(const x:single):single;inline;
function get_activation(s:string):TActivation;

function get_activation_string(const a:TActivation):string;

function activate(const x:single; const a:TActivation):single;

function gradient(const x:single; const a:TActivation):single;

procedure gradient_array(const x:PSingle; const n:longint; const a:TActivation; const delta:PSingle);   overload;
procedure activate_array(const x:PSingle; const n:longint;const a:TActivation);                         overload;

procedure gradient_array_swish(const x: Psingle; const n: longint; const sigmoid: Psingle; delta: Psingle);
procedure gradient_array_mish(const n: longint; const activation_input: Psingle; delta: Psingle);
procedure gradient_array_hard_mish(const n: longint; const activation_input, delta: Psingle);
procedure activate_array_swish(const x: Psingle; const n: longint; output_sigmoid, output: Psingle);
procedure activate_array_mish(const x: Psingle; const n: longint; const activation_input, output: Psingle);
procedure activate_array_hard_mish(const x: Psingle; const n: longint; const activation_input, output: PSingle);
procedure activate_array_normalize_channels(const x: Psingle; const n, batch, channels, wh_step:longint; const output: Psingle);
procedure activate_array_normalize_channels_softmax(const x: Psingle; const n, batch, channels, wh_step: longint; const output: PSingle; const use_max_val: boolean);
procedure gradient_array_normalize_channels_softmax(const x: Psingle; const n, batch, channels, wh_step: longint; const delta: Psingle);
procedure gradient_array_normalize_channels(const x: Psingle; const n, batch, channels, wh_step: longint; const delta: Psingle);

//procedure gradient_array(const x:TSingles; const n:longint; const a:TActivation; const delta:TSingles); overload;
//procedure activate_array(const x:TSingles; const n:longint;const a:TActivation);                        overload;

{$ifdef GPU}
procedure activate_array_gpu(const x:PSingle; const n:longint; const a:TActivation);

procedure gradient_array_gpu(const x:PSingle; const n:longint; const a:TActivation; const delta:PSingle);

{$endif}


implementation
uses steroids, blas;

// todo SIMDfy the activations




//FastExpSse(float __vector(4)):
//  mulps xmm0, XMMWORD PTR .LC4[rip]
//  cvtps2dq xmm0, xmm0
//  paddd xmm0, XMMWORD PTR .LC5[rip]
//  ret
//.LC4:
//  .long 1262004795
//  .long 1262004795
//  .long 1262004795
//  .long 1262004795
//.LC5:
//  .long 1065054451
//  .long 1065054451
//  .long 1065054451
//  .long 1065054451

function  stair_activate(const x:single) :single;inline;
var n :longint;
begin

  n := floor(x);
  if n mod 2 = 0 then
    exit(floor(x/ 2));
  exit((x - n) + floor(x/2));

end;

function  hardtan_activate(const x:single):single;inline;
begin

    if x < -1 then exit(-1);
    if x > 1 then exit(1);
    exit(x);

end;

function linear_activate(const x:single):single;inline;
begin
  result:=x;
end;

{$if defined(CPUX64) and defined(FPUAVX2)}
procedure logistic_array(const dst, src:PSingle; const N:PtrInt);
const
  l2e :single = 1.442695041;// log2(e);
  c0  :single = 1.00172476;
  c1  :single = 0.657636276;
  c2  :single = 0.3371894346;
  //MAX_EXP =  8.8722839052068352E+001;
  //MIN_EXP = -8.7336544750553102E+001;

  MAX_EXP =  8.87E+001;
  MIN_EXP = -8.73E+001;

  one :array[0..7] of single = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
  zero:array[0..7] of single = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  mx  :array[0..7] of single = (MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP);
  mn  :array[0..7] of single = (MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP );
asm
  sub                  rsp      , 16*2                     // making stack space to save one xmm size register
  vmovdqu              [rsp+$00], xmm6
  vmovdqu              [rsp+$10], xmm7

  vpbroadcastd  ymm3  , [rip + l2e]
  vpbroadcastd  ymm4  , [rip + c1]
  vpbroadcastd  ymm5  , [rip + c0]

  mov           r11   , N
  shr           r11   , 3
  jz            @rem

@while:
  vxorps        ymm0  , ymm0        , ymm0              // zero
  vsubps        ymm1  , ymm0        , [src]             // -src
  vcmpgeps      ymm6  , ymm1        , [rip + mx]
  vcmpleps      ymm7  , ymm1        , [rip + mn]
  vblendvps     ymm1  , ymm1 , [rip + mx], ymm6
  vblendvps     ymm1  , ymm1 , [rip + mn], ymm7
  vmulps        ymm1  , ymm3        , ymm1
  vroundps      ymm2  , ymm1        , 1
  vsubps        ymm1  , ymm1        , ymm2
  vcvtps2dq     ymm0  , ymm2
  vpbroadcastd  ymm2  , [rip + c2]
  vfmadd213ps   ymm2  , ymm1        , ymm4
  vpslld        ymm0  , ymm0        , 23
  vfmadd213ps   ymm1  , ymm2        , ymm5
  vpaddd        ymm0  , ymm0        , ymm1
  vaddps        ymm1  , ymm0        , [rip + one]       // 1 +exp(-src)
  vrcpps        ymm1  , ymm1                            // 1/(1+exp(-src))
  //vpcmpeqd      ymm0  , ymm0        , ymm0              // set ymm0 to to 0xffffffff
  //vpandn        ymm6  , ymm6        , ymm0              // ymm6 := not ymm6
  //vpandn        ymm7  , ymm7        , ymm0              // ymm6 := not ymm6
  //vblendvps     ymm1  , ymm1  , [rip+one] , ymm6
  //vblendvps     ymm1  , ymm1  , [rip+zero] , ymm7
  vmovups       [dst] , ymm1
  add           src   , 32
  add           dst   , 32
  dec           r11
  jnz           @while

  and           N   , 7
  jz            @done
@rem:

  vpxor         xmm0  , xmm0        , xmm0
  vsubss        xmm1  , xmm0        , dword [src]              //-src
  vcmpgeps      xmm6  , xmm1        , [rip + mx]
  vcmpleps      xmm7  , xmm1        , [rip + mn]
  vblendvps     xmm1  , xmm1 , [rip + mx], xmm6
  vblendvps     xmm1  , xmm1 , [rip + mn], xmm7
  vmulss        xmm1  , xmm3        , xmm1
  roundss       xmm2  , xmm1        , 1
  vsubss        xmm1  , xmm1        , xmm2
  vcvtps2dq     xmm0  , xmm2
  vmovss        xmm2  , [rip + c2]
  vfmadd213ss   xmm2  , xmm1        , xmm4
  vpslld        xmm0  , xmm0        , 23
  vfmadd213ss   xmm1  , xmm2        , xmm5
  vpaddd        xmm0  , xmm0        , xmm1
  vaddss        xmm1  , xmm0        , [rip + one]       // 1 +exp(-src)
  vrcpss        xmm1  , xmm1        , xmm1              // 1/(1+exp(-src))
  //vpcmpeqd      xmm0  , xmm0        , xmm0              // set ymm0 to to 0xffffffff
  //vpandn        xmm6  , xmm6        , xmm0              // ymm6 := not ymm6
  //vpandn        xmm7  , xmm7        , xmm0              // ymm6 := not ymm6
  //vblendvps     xmm1  , xmm1  , [rip+one] , xmm6
  //vblendvps     xmm1  , xmm1  , [rip+zero]  , xmm7
  vmovss        dword [dst] , xmm1
  add           src   , 4
  add           dst   , 4
  dec           N
  jnz           @rem
@done:
  vmovdqu              xmm6     , [rsp+$00]
  vmovdqu              xmm7     , [rsp+$10]
  add                  rsp      , 16*2                     // restoring stack
end;

procedure SiLU_array(const dst, sigmoid, src:PSingle; const N:PtrInt);
const
  l2e :single = 1.442695041;// log2(e);
  c0  :single = 1.00172476;
  c1  :single = 0.657636276;
  c2  :single = 0.3371894346;
  //MAX_EXP =  8.8722839052068352E+001;
  //MIN_EXP = -8.7336544750553102E+001;

  MAX_EXP =  8.87E+001;
  MIN_EXP = -8.73E+001;

  one :array[0..7] of single = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
  zero:array[0..7] of single = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  mx  :array[0..7] of single = (MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP, MAX_EXP);
  mn  :array[0..7] of single = (MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP, MIN_EXP );
asm
  sub                  rsp      , 16*2                     // making stack space to save one xmm size register
  vmovdqu              [rsp+$00], xmm6
  vmovdqu              [rsp+$10], xmm7

  vpbroadcastd  ymm3  , [rip + l2e]
  vpbroadcastd  ymm4  , [rip + c1]
  vpbroadcastd  ymm5  , [rip + c0]

  mov           r11   , N
  shr           r11   , 3
  jz            @rem

@while:
  vxorps        ymm0  , ymm0        , ymm0              // zero
  vsubps        ymm1  , ymm0        , [src]             // -src
  vcmpgeps      ymm6  , ymm1        , [rip + mx]
  vcmpleps      ymm7  , ymm1        , [rip + mn]
  vblendvps     ymm1  , ymm1 , [rip + mx], ymm6
  vblendvps     ymm1  , ymm1 , [rip + mn], ymm7
  vmulps        ymm1  , ymm3        , ymm1
  vroundps      ymm2  , ymm1        , 1
  vsubps        ymm1  , ymm1        , ymm2
  vcvtps2dq     ymm0  , ymm2
  vpbroadcastd  ymm2  , [rip + c2]
  vfmadd213ps   ymm2  , ymm1        , ymm4
  vpslld        ymm0  , ymm0        , 23
  vfmadd213ps   ymm1  , ymm2        , ymm5
  vpaddd        ymm0  , ymm0        , ymm1
  vaddps        ymm1  , ymm0        , [rip + one]       // 1 +exp(-src)
  vrcpps        ymm1  , ymm1                            // 1/(1+exp(-src))
  //vpcmpeqd      ymm0  , ymm0        , ymm0              // set ymm0 to to 0xffffffff
  //vpandn        ymm6  , ymm6        , ymm0              // ymm6 := not ymm6
  //vpandn        ymm7  , ymm7        , ymm0              // ymm6 := not ymm6
  //vblendvps     ymm1  , ymm1  , [rip+one] , ymm6
  //vblendvps     ymm1  , ymm1  , [rip+zero] , ymm7
  vmovups       [sigmoid] , ymm1
  vmulps        ymm1      , ymm1    , [src]
  vmovups       [dst]     , ymm1
  add           src       , 32
  add           sigmoid   , 32
  add           dst       , 32
  dec           r11
  jnz           @while

  and           N   , 7
  jz            @done
@rem:

  vpxor         xmm0  , xmm0        , xmm0
  vsubss        xmm1  , xmm0        , dword [src]              //-src
  vcmpgeps      xmm6  , xmm1        , [rip + mx]
  vcmpleps      xmm7  , xmm1        , [rip + mn]
  vblendvps     xmm1  , xmm1 , [rip + mx], xmm6
  vblendvps     xmm1  , xmm1 , [rip + mn], xmm7
  vmulss        xmm1  , xmm3        , xmm1
  roundss       xmm2  , xmm1        , 1
  vsubss        xmm1  , xmm1        , xmm2
  vcvtps2dq     xmm0  , xmm2
  vmovss        xmm2  , [rip + c2]
  vfmadd213ss   xmm2  , xmm1        , xmm4
  vpslld        xmm0  , xmm0        , 23
  vfmadd213ss   xmm1  , xmm2        , xmm5
  vpaddd        xmm0  , xmm0        , xmm1
  vaddss        xmm1  , xmm0        , [rip + one]       // 1 +exp(-src)
  vrcpss        xmm1  , xmm1        , xmm1              // 1/(1+exp(-src))
  //vpcmpeqd      xmm0  , xmm0        , xmm0              // set ymm0 to to 0xffffffff
  //vpandn        xmm6  , xmm6        , xmm0              // ymm6 := not ymm6
  //vpandn        xmm7  , xmm7        , xmm0              // ymm6 := not ymm6
  //vblendvps     xmm1  , xmm1  , [rip+one] , xmm6
  //vblendvps     xmm1  , xmm1  , [rip+zero]  , xmm7
  vmovss               dword [sigmoid] , xmm1
  vmulss               xmm1            , xmm1 ,   dword [src]
  vmovss               [dst]           , xmm1
  add                  src             , 4
  add                  sigmoid         , 4
  add                  dst             , 4
  dec                  N
  jnz                  @rem
@done:
  vmovdqu              xmm6            , [rsp+$00]
  vmovdqu              xmm7            , [rsp+$10]
  add                  rsp             , 16*2                     // restoring stack
end;

{$else}

procedure logistic_array(const dst,src:PSingle; const N:longint);inline;
var i:longint;
begin
  for i:=0 to N-1 do
      dst[i] := 1/(1 + exp(-src[i]))
end;
{$endif}


function logistic_activate(const x:single):single;//inline;
begin
  result := 1/(1 + exp(-x))
end;

function loggy_activate(const x:single):single;inline;
begin
  result := 2/(1 + exp(-x)) - 1;
end;

function relu_activate(const x:single):single;inline;
begin
  result := x*longint(x>0);
  //if x<0 then exit(0);
  //result:=x
end;

function relu6_activate(const x:single):single;inline;
begin
  //min_val_cmp(max_val_cmp(x, 0), 6)
  //result := EnsureRange(x,0,6);
  result:= x*longint(x>0) * longint(x<=6)
end;

function elu_activate(const x:single):single;inline;
begin
  result:=longint(x >= 0)*x + longint(x < 0)*(exp(x)-1);
end;

function selu_activate(const x:single):single;inline;
begin
  result:= longint(x >= 0)*1.0507*x + longint(x < 0)*1.0507*1.6732*(exp(x)-1);
end;

function gelu_activate(const x:single):single;inline;
begin
  result:= 0.5*x*(1 + tanh(0.797885*x + 0.035677*power(x, 3)));
end;

function relie_activate(const x:single):single;inline;
begin
  if x>0 then result:=x
  else result:= 0.01*x;
end;

function ramp_activate(const x:single):single;inline;
begin
  result:= x*longint(x>0)+0.1*x;
end;

function leaky_activate(const x:single):single;inline;
begin
  if x>0 then result:= x
  else result := 0.1*x
end;

{$if defined(CPUX64) and defined(FPUAVX2)}
procedure leaky_array(const x:PSingle; const N:longint);assembler;{$ifdef FPC}nostackframe;{$endif}
const f:single=0.1;
asm
  vxorps            ymm0  , ymm0  ,  ymm0
  movss             xmm1  , [rip+f]
  vbroadcastss      ymm1  , xmm1

  mov               r11d   , N
  shr               r11   , 3                    // N div 8
  jz                @rem                         // goto rem if zero

@while:
  vcmpgtps          ymm2  , ymm0  ,  [x]         //  is 0 < x
  vmulps            ymm3  , ymm1  ,  [x]
  vmaskmovps        yword [x]  , ymm2  ,  ymm3
  add               x     , 8*4                  // next 8 packed singles
  dec               r11
  jnz               @while                       // while r11<>0

@rem:
  and               N     , 7                    // N mod 8
  jz                @done                        // exit if zero

@while2:
  vcomiss           xmm0  , dword [x]
  jbe               @skip
  vmulss            xmm3  , xmm1  ,  dword [x]
  vmovss            dword [x]   , xmm3
@skip:
  add               x     , 4
  dec               N
  jnz               @while2

@done:
end;

{$else}
procedure leaky_array(const x:psingle; const N:longint);
var i:longint;
begin
  for i:=0 to N-1 do
      x[i]:=(longint(x[i]>0) + longint(x[i]<0)*0.1)*x[i]
end;
{$endif}


function tanh_activate(const x:single):single;inline;
begin
  result := 2 / (1 + exp(-2 * x)) - 1
  //result:= (exp(2*x)-1)/(exp(2*x)+1);
end;

function softplus_activate(const x, threshold : single):single;inline;
begin
    if x > threshold then
      exit(x)                // too large
    else if x < -threshold then
      exit(exp(x));    // too small
    exit(ln(exp(x) + 1));
    //exit(LnXP1(x));
end;

function plse_activate(const x:single):single;inline;
begin
    if x < -4 then exit( 0.01 * (x + 4));
    if x > 4 then exit( 0.01 * (x - 4) + 1);
    result := 0.125*x + 0.5
end;


function lhtan_activate(const x:single):single;inline;
begin
    if(x < 0) then exit(0.001*x);
    if(x > 1) then exit(0.001*(x-1) + 1);
    result := x
end;

function lhtan_gradient(const x:single):single;inline;
begin
    if (x > 0) and  (x < 1) then
      exit(1);
    exit(0.001)
end;


function hardtan_gradient(const x:single):single;inline;
begin
    if (x > -1) and (x < 1) then
      exit(1);
    exit(0);
end;

function linear_gradient(const x:single):single;inline;
begin
  result:= 1;
end;

function logistic_gradient(const x:single):single;inline;
begin
  result := (1-x)*x;
end;

function loggy_gradient(const x:single):single;inline;
var y:single;
begin
    y := (x+1.0)/2.0;
    result:= 2*(1-y)*y;
end;

function stair_gradient(const x:single):single;inline;
begin
    if floor(x) = x then exit( 0);
    result := 1;
end;

function relu_gradient(const x:single):single;inline;
begin
  result := longint(x>0);
end;

function relu6_gradient(const x:single):single;inline;
begin
  result := longint((x>0) and (x<6));
end;

function elu_gradient(const x:single):single;inline;
begin
  result := longint(x >= 0) + longint(x < 0)*(x + 1);
end;

function selu_gradient(const x:single):single;inline;
begin
  result := longint(x >= 0)*1.0507 + longint(x < 0)*(x + 1.0507*1.6732);
end;

function relie_gradient(const x:single):single;inline;
begin
  if x>0 then result := 1
  else result := 0.01
end;

function ramp_gradient(const x:single):single;inline;
begin
  result := longint(x>0) + 0.1;
end;

function leaky_gradient(const x:single):single;inline;
begin
  if x>0 then result := 1
  else result := 0.1;
end;

function tanh_gradient(const x:single):single;inline;
begin
  result := 1-x*x;
end;

function sech(const x:single):single;inline;
begin
    result := 2 / (exp(x) + exp(-x))
end;

function gelu_gradient(const x:single):single;inline;
var x3 : single;
begin
    x3 := power(x,3);
    result := 0.5*tanh(0.0356774*x3 + 0.797885*x) + (0.0535161*x3 + 0.398942*x) * power(sech(0.0356774*x3 + 0.797885*x), 2) + 0.5
end;

function plse_gradient(const x:single):single;inline;
begin
  if (x < 0) or (x > 1) then
    result :=  0.01
  else
    result := 0.125;
end;


function get_activation_string(const a: TActivation): string;
begin
  //case a of
  //    acLOGISTIC:
  //        result := 'logistic';
  //    acLOGGY:
  //        result := 'loggy';
  //    acRELU:
  //        result := 'relu';
  //    acELU:
  //        result := 'elu';
  //    acSELU:
  //        result := 'selu';
  //    acRELIE:
  //        result := 'relie';
  //    acRAMP:
  //        result := 'ramp';
  //    acLINEAR:
  //        result := 'linear';
  //    acTANH:
  //        result := 'tanh';
  //    acPLSE:
  //        result := 'plse';
  //    acLEAKY:
  //        result := 'leaky';
  //    acSTAIR:
  //        result := 'stair';
  //    acHARDTAN:
  //        result := 'hardtan';
  //    acLHTAN:
  //        result := 'lhtan';
  //    else
          result := LowerCase(copy(GetEnumName(TypeInfo(TActivation), ord(a)),3))// result:='relu'
  //end;
end;

function get_activation(s: string): TActivation;
begin
    s := LowerCase(s);
    if s = 'logistic' then  exit(acLOGISTIC);
    if s = 'swish' then exit(acSWISH);
    if s = 'mish' then exit(acMISH);
    if s = 'hard_mish' then exit(acHARD_MISH);
    if s = 'normalize_channels' then exit(acNORM_CHAN);
    if s = 'normalize_channels_softmax' then exit(acNORM_CHAN_SOFTMAX);
    if s = 'normalize_channels_softmax_maxval' then exit(acNORM_CHAN_SOFTMAX_MAXVAL);
    if s = 'loggy' then  exit(acLOGGY);
    if s = 'relu' then  exit(acRELU);
    if s = 'relu6' then  exit(acRELU6);
    if s = 'elu' then  exit(acELU);
    if s = 'selu' then  exit(acSELU);
    if s = 'gelu' then  exit(acGELU);
    if s = 'relie' then  exit(acRELIE);
    if s = 'plse' then  exit(acPLSE);
    if s = 'hardtan' then  exit(acHARDTAN);
    if s = 'lhtan' then  exit(acLHTAN);
    if s = 'linear' then  exit(acLINEAR);
    if s = 'ramp' then  exit(acRAMP);
    if s = 'revleaky' then  exit(acREVLEAKY);
    if s = 'leaky' then  exit(acLEAKY);
    if s = 'tanh' then  exit(acTANH);
    if s = 'stair' then  exit(acSTAIR);
    writeln('Couldn''t find activation function ',s,' going with ReLU');
    exit(acRELU)
end;

// todo SIMD [activate] function
// note perhapse assigning the function is better than a switch case
function activate(const x:single ;const a:TActivation):single;
begin
    case a of
      acLINEAR:
          result := linear_activate(x);
      acLOGISTIC:
          result := logistic_activate(x);
      acLOGGY:
          result := loggy_activate(x);
      acRELU:
          result := relu_activate(x);
      acELU:
          result := elu_activate(x);
      acSELU:
          result := selu_activate(x);
      acGELU:
          result := gelu_activate(x);
      acRELIE:
          result := relie_activate(x);
      acRAMP:
          result := ramp_activate(x);
      acLEAKY, acREVLEAKY:
          result := leaky_activate(x);
      acTANH:
          result := tanh_activate(x);
      acPLSE:
          result := plse_activate(x);
      acSTAIR:
          result := stair_activate(x);
      acHARDTAN:
          result := hardtan_activate(x);
      acLHTAN:
          result := lhtan_activate(x);
      else
          result := 0
    end;
end;

// todo SIMD activate_array

procedure logistic(const f,t :PtrInt; const p:pointer=nil);
var i : PtrInt;
   a:PMPParams absolute p;
   x:Psingle;
begin
  x:=a.A;
  //{$if defined(CPUX64) and defined(AVX2)}
  inc(x, f);
  logistic_array(x, x, t-f+1);
  //{$else}
  //for i:=f to t do
  //    x[i] := logistic_activate(x[i]);
  //{$endif}
end;

procedure leaky(const f,t:PtrInt; const p:pointer=nil);
var
   a:PMPParams absolute p;
   x:PSingle;
begin
  x:=a.A;
  leaky_array(x + f,t-f+1);
end;


procedure activate_array(const x: PSingle; const n: longint;
  const a: TActivation);
var i:longint;
   p:TMPParams;
begin
  {$ifdef USE_TELEMETRY} if benchmark then metrics.act.start(a);{$endif}

  p.A:=x;
  case a of
    acLEAKY:
      {$if defined(_USE_MULTITHREADING)}
        mp2.&for(leaky,0,N-1,@p);
      {$else}
        leaky(0, N-1, @p);
      {$endif}
    acLINEAR:;

    acLOGISTIC:
      {$if defined(USE_MULTITHREADING)}
        mp2.&For(logistic, 0, n-1,@p);
      {$else}
        logistic(0, N-1, @p);
      {$endif}
//        for i := 0 to n-1 do
//            x[i] := logistic_activate(x[i]);
    else
        for i := 0 to n-1 do
            x[i] := activate(x[i], a)
  end ;
  {$ifdef USE_TELEMETRY} if benchmark then metrics.act.finish(a);{$endif}
end;

  procedure swishMP(const f, t:PtrInt; const p:pointer);
  var
    i: longint;
    x_val, sigmoid: single;
    a:PMPParams absolute p;
    x, output_sigmoid, output:PSingle;
  begin
      x:=a.A;
      output_sigmoid:=a.B;
      output := a.C;

      {$if defined(CPUX64) and defined(FPUAVX2)}
      SiLU_array(output+f, output_sigmoid+f, x+f, t-f+1);
      {$else}
      for i := f to t do
        begin
            x_val := x[i];
            sigmoid := logistic_activate(x_val);
            output_sigmoid[i] := sigmoid;
            output[i] := x_val * sigmoid
        end
      {$endif}
  end;

// SWISH aka SiLU
procedure activate_array_swish(const x: Psingle; const n: longint;
  output_sigmoid, output: Psingle);
var
  p:TMPParams;
begin
  // todo simdfy
  {$ifdef USE_TELEMETRY} if benchmark then metrics.act.start(acSWISH);{$endif}
  p.A:=x;
  p.B:=output_sigmoid;
  p.C:=output;
  {$if defined(USE_MULTITHREADING)}
  mp2.&for(swishMP, 0, n-1,@p);
  {$else}
  swishMP(0, N-1, @p);
  {$endif}
  {$ifdef USE_TELEMETRY}if benchmark then metrics.act.finish(acSWISH);{$endif}
end;

  procedure mishMP(const f,t :PtrInt;const p:pointer=nil);
  const MISH_THRESHOLD : single=20;
  var
    i: longint;
    x_val: single;
    a:PMPParams absolute p;
    x, activation_input, output:Psingle;
  begin
      x                 :=a.A;
      activation_input  :=a.B;
      output            :=a.C;

      for i := f to t do
        begin
            x_val := x[i];
            activation_input[i] := x_val;
            output[i] := x_val * tanh_activate(softplus_activate(x_val, MISH_THRESHOLD))
        end
  end;

procedure activate_array_mish(const x: Psingle; const n: longint;
  const activation_input, output: Psingle);
var
  p:TMPParams;
begin
  {$ifdef USE_TELEMETRY} if benchmark then metrics.act.start(acMISH);{$endif}
  p.A:=x;
  p.B:=activation_input;
  p.C:=output;
  // todo SIMDfy
  {$if defined(USE_MULTITHREADING)}
  mp2.&for(mishMP, 0, n-1,@p)
  {$else}
  mishMP(0, N-1, @p)
  {$endif}
  ;
  {$ifdef USE_TELEMETRY} if benchmark then metrics.act.finish(acMISH);{$endif}
end;


function hard_mish_yashas(x: single):single;
begin
    if (x > 0) then
        exit(x);
    if x > -2 then
        exit(x * x / 2+x);
    exit(0)
end;

procedure activate_array_hard_mish(const x: Psingle; const n: longint;
  const activation_input, output: PSingle);
var
    i: longint;
    x_val: single;
begin
  // todo SIMDfy
    {$ifdef USE_TELEMETRY} if benchmark then metrics.act.start(acHARD_MISH);{$endif}
    for i := 0 to n -1 do
        begin
            x_val := x[i];
            activation_input[i] := x_val;
            output[i] := hard_mish_yashas(x_val)
        end;
    {$ifdef USE_TELEMETRY} if benchmark then metrics.act.finish(acHARD_MISH);{$endif}
end;

procedure activate_array_normalize_channels(const x: Psingle; const n, batch,
  channels, wh_step: longint; const output: Psingle);
const eps: single = 0.0001;
var
    size, i, wh_i, b, k: longint;
    sum, val: single;
begin
    {$ifdef USE_TELEMETRY} if benchmark then metrics.act.start(acNORM_CHAN);{$endif}
    size := n div channels;
    // todo SIMDfy
    for i := 0 to size -1 do
        begin
            wh_i := i mod wh_step;
            b := i div wh_step;
            if i < size then
                begin
                    sum := eps;
                    for k := 0 to channels -1 do
                        begin
                            val := x[wh_i+k * wh_step+b * wh_step * channels];
                            if val > 0 then
                                sum := sum + val
                        end;
                    for k := 0 to channels -1 do
                        begin
                            val := x[wh_i+k * wh_step+b * wh_step * channels];
                            if val > 0 then
                                val := val / sum
                            else
                                val := 0;
                            output[wh_i+k * wh_step+b * wh_step * channels] := val
                        end
                end
        end;
    {$ifdef USE_TELEMETRY} if benchmark then metrics.act.finish(acNORM_CHAN);{$endif}
end;

procedure activate_array_normalize_channels_softmax(const x: Psingle; const n,
  batch, channels, wh_step: longint; const output: PSingle;
  const use_max_val: boolean);
const eps: single = 0.0001;
var
    size, i, wh_i, b, k: longint;
    sum, max_val, val: single;
begin
    {$ifdef USE_TELEMETRY} if benchmark then metrics.act.start(acNORM_CHAN_SOFTMAX);{$endif}
    size := n div channels;
    // todo SIMDFy
    for i := 0 to size -1 do
        begin
            wh_i := i mod wh_step;
            b := i div wh_step;
            if i < size then
                begin
                    sum := eps;
                    max_val := -MaxSingle;
                    if use_max_val then
                        for k := 0 to channels -1 do
                            begin
                                val := x[wh_i+k * wh_step+b * wh_step * channels];
                                if (val > max_val) or (k = 0) then
                                    max_val := val
                            end
                    else
                        max_val := 0;
                    for k := 0 to channels -1 do
                        begin
                            val := x[wh_i+k * wh_step+b * wh_step * channels];
                            sum := sum + exp(val-max_val)
                        end;
                    for k := 0 to channels -1 do
                        begin
                            val := x[wh_i+k * wh_step+b * wh_step * channels];
                            val := exp(val-max_val) / sum;
                            output[wh_i+k * wh_step+b * wh_step * channels] := val
                        end
                end
        end;
    {$ifdef USE_TELEMETRY} if benchmark then metrics.act.finish(acNORM_CHAN_SOFTMAX);{$endif}
end;

procedure gradient_array_normalize_channels_softmax(const x: Psingle; const n,
  batch, channels, wh_step: longint; const delta: Psingle);
var
    size, i, wh_i, b, k, index: longint;
    grad, &out, d: single;
begin
    {$ifdef USE_TELEMETRY} if benchmark then metrics.grad.start(acNORM_CHAN_SOFTMAX);{$endif}
    size := n div channels;
    // todo SIMDfy
    for i := 0 to size -1 do
        begin
            wh_i := i mod wh_step;
            b := i div wh_step;
            if i < size then
                begin
                    grad := 0;
                    for k := 0 to channels -1 do
                        begin
                            index := wh_i+k * wh_step+b * wh_step * channels;
                            &out := x[index];
                            d := delta[index];
                            grad := grad + (&out * d)
                        end;
                    for k := 0 to channels -1 do
                        begin
                            index := wh_i+k * wh_step+b * wh_step * channels;
                            d := delta[index];
                            d := d * grad;
                            delta[index] := d
                        end
                end
        end;
    {$ifdef USE_TELEMETRY} if benchmark then metrics.grad.finish(acNORM_CHAN_SOFTMAX);{$endif}
end;

procedure gradient_array_normalize_channels(const x: Psingle; const n, batch,
  channels, wh_step: longint; const delta: Psingle);
var
    size, i, wh_i, b, k, index: longint;
    grad, &out, d: single;
begin
    {$ifdef USE_TELEMETRY} if benchmark then metrics.grad.start(acNORM_CHAN);{$endif}
    size := n div channels;
    // todo SIMDfy
    for i := 0 to size -1 do
        begin
            wh_i := i mod wh_step;
            b := i div wh_step;
            if i < size then
                begin
                    grad := 0;
                    for k := 0 to channels -1 do
                        begin
                            index := wh_i+k * wh_step+b * wh_step * channels;
                            &out := x[index];
                            d := delta[index];
                            grad := grad + (&out * d)
                        end;
                    for k := 0 to channels -1 do
                        begin
                            index := wh_i+k * wh_step+b * wh_step * channels;
                            if x[index] > 0 then
                                begin
                                    d := delta[index];
                                    d := d * grad;
                                    delta[index] := d
                                end
                        end
                end
        end;
    {$ifdef USE_TELEMETRY} if benchmark then metrics.grad.finish(acNORM_CHAN);{$endif}
end;



function gradient(const x:single; const a: TActivation):single;
begin
    case a of
        acLINEAR:
            result := linear_gradient(x);
        acLOGISTIC:
            result := logistic_gradient(x);
        acLOGGY:
            result := loggy_gradient(x);
        acRELU:
            result := relu_gradient(x);
        acRELU6:
            result := relu6_gradient(x);
        acNORM_CHAN, acNORM_CHAN_SOFTMAX, acNORM_CHAN_SOFTMAX_MAXVAL:
            raise Exception.Create('Error: should be used custom NORM_CHAN or NORM_CHAN_SOFTMAX-function for gradien');
        acELU:
            result := elu_gradient(x);
        acSELU:
            result := selu_gradient(x);
        acRELIE:
            result := relie_gradient(x);
        acRAMP:
            result := ramp_gradient(x);
        acLEAKY:
            result := leaky_gradient(x);
        acTANH:
            result := tanh_gradient(x);
        acPLSE:
            result := plse_gradient(x);
        acSTAIR:
            result := stair_gradient(x);
        acHARDTAN:
            result := hardtan_gradient(x);
        acLHTAN:
            result := lhtan_gradient(x);
        else
            result := 0;
    end;
end;

// todo SIMD gradient_array
procedure gradient_array(const x:PSingle; const n:longint; const a:TActivation; const delta:PSingle);
var i:longint;
begin
    {$ifdef USE_TELEMETRY} if benchmark then metrics.grad.start(a);{$endif}
    // todo SIMDfy
    for i := 0 to n-1 do
        delta[i] := delta[i] * gradient(x[i], a);
    {$ifdef USE_TELEMETRY} if benchmark then metrics.grad.finish(a);{$endif}
end;

procedure gradient_array_swish(const x: Psingle; const n: longint; const sigmoid: Psingle; delta: Psingle);
var
    i: longint;
    swish: single;
begin
    {$ifdef USE_TELEMETRY} if benchmark then metrics.grad.start(acSWISH);{$endif}
    // todo SIMDfy
    for i := 0 to n -1 do
        begin
            swish := x[i];
            delta[i] := delta[i] * (swish+sigmoid[i] * (1-swish))
        end;
    {$ifdef USE_TELEMETRY} if benchmark then metrics.grad.finish(acSWISH);{$endif}
end;

procedure gradient_array_mish(const n: longint; const activation_input: Psingle; delta: Psingle);
const
    MISH_THRESHOLD: single = 20;
var
    i: longint;
    inp, sp, grad_sp, tsp, grad_tsp, grad: single;
begin
    {$ifdef USE_TELEMETRY} if benchmark then metrics.grad.start(acMISH);{$endif}

    // todo SIMDfy
    for i := 0 to n -1 do
        begin
            inp := activation_input[i];
            sp := softplus_activate(inp, MISH_THRESHOLD);
            grad_sp := 1-exp(-sp);
            tsp := tanh(sp);
            grad_tsp := (1-tsp * tsp) * grad_sp;
            grad := inp * grad_tsp+tsp;
            delta[i] := delta[i] * grad
        end ;
    {$ifdef USE_TELEMETRY} if benchmark then metrics.grad.finish(acMISH);{$endif}
end;

function hard_mish_yashas_grad(x: single):single;
begin
    if (x > 0) then
        exit(1);
    if x > -2 then
        exit(x+1);
    exit(0)
end;

procedure gradient_array_hard_mish(const n: longint; const activation_input,
  delta: Psingle);
var
    i: longint;
    inp: single;
begin
    {$ifdef USE_TELEMETRY} if benchmark then metrics.grad.start(acHARD_MISH);{$endif}
    // todo SIMDfy
    for i := 0 to n -1 do
        begin
            inp := activation_input[i];
            delta[i] := delta[i] * hard_mish_yashas_grad(inp)
        end;
    {$ifdef USE_TELEMETRY} if benchmark then metrics.grad.finish(acHARD_MISH);{$endif}
end;


//procedure gradient_array(const x: TSingles; const n: longint; const a: TActivation; const delta: TSingles);
//begin
//  if length(x)=0 then exit;
//  gradient_array(PSingle(@x[0]), n, a, PSingle(@delta[0]));
//end;
//
//procedure activate_array(const x: TSingles; const n: longint; const a: TActivation);
//begin
//  if length(x)=0 then exit;
//  activate_array(PSingle(@x[0]), n, a);
//end;
const N=299;EPS=0.19;
var
    a, b, c:TArray<single>;
    diff,f1,f2,f3,f4:double;
    i: longint;
initialization



  //setLength(a,N+N+1);
  //setLength(b,Length(a));
  //setLength(c,length(a));
  //
  //for i:=0 to high(a) do
  //  a[i]:=i-N ;
  //
  //
  //logistic_array(@c[0],@a[0], length(a));
  //
  //for i:=0 to high(a) do begin
  //      b[i] := logistic_activate(a[i])
  //end;
  //
  //write('      ','id':3, ': B');write(' ~ C');write('id':3, ': ');writeln('B ~ C');
  //for i:=0 to high(a) do
  //begin
  //  if i mod 2 > 0 then begin
  //      write('      ',i-N:3, ': ',b[i]);
  //      f1:=c[i];
  //      f2:=b[i];
  //      if f2<>0 then
  //        diff:=100*abs(f1-f2) / f2
  //      else
  //        diff:=0;
  //      write(' ~ ',c[i]);writeln('  ',diff:3:3,'% ',boolToStr(diff<eps,'T','F'))
  //  end
  //  else begin
  //      write(i-N:3, ': ');
  //      f1:=c[i];
  //      f2:=b[i];
  //      if f2<>0 then
  //        diff:=100*abs(f1-f2) / f2
  //      else
  //        diff:=0;
  //      write(b[i],' ~ ',c[i]);write('  ',diff:3:3,'% ',boolToStr(diff<eps,'T','F'),'     ')
  //  end
  //end;
  //readln();
//  //halt(0)

end.

