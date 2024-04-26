function [handle]=atldk(varargin)
% visualizza tracce sismiche
% usi tipici:
%  atldk(time,offset,data);
%  atldk(data); in campioni
%  [h]=atldk... ritorna gli handles degli oggetti di tipo "line"
% personalizza utilizzando coppie di argomenti:
%  atldk(...,'parameter',value,...,...);
%  i parametri ed i possibili valori sono:
%  'dir' , 'vertical'|'v'|'vert'|'ver'|'horizontal'|'h'|'hor';
%  default direction = 'vertical'
%  'color' , 'k'|'w'|'g'|'r'|'y'|'m'|'c'|'b'|[red,green,blue];
%  default color = 'k';
%  'marker' , 'none'|'.'|'o'|'x'|'+'|'*'|'s'|'d'|'v'|'^'|'<'|'>'|'p'|'h';
%  default marker= 'none'
%  'linestyle' , '-'|':'|'-.'|'--'
%  default linestyle = '-';
%  'amp' , 'entire'|'single'|'raw','agc';
%  default amplitude mode = 'entire';
%  'gain' , value;
%  default gain = 1;
%  'wiggle' , 'none'|'accurate'|'fast';
%  default wiggle mode = 'none';
%  'clip' , 'none'| value;
%  default clip mode = 'none';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%interpretazione dell'input----------------------------------------------------

charindx=zeros(1,nargin);
for I=1:nargin
   charindx(I)=isa(varargin{I},'char');
end
charmin=min(find(charindx));
if isempty(charmin)
   charmin=[nargin+1];
end

nummin=charmin-1;

if nummin<1 | nummin>3
   error(' atldk: input error')
end

if nummin==1;     % non sono forniti gli assi
   in=varargin{1};
   X=[1:size(in,1)];
   Y=[1:size(in,2)];
elseif nummin==3; % sono forniti gli assi
   in=varargin{3};
   X=varargin{1};
   Y=varargin{2};
elseif nummin==2; % fornito solo asse X
   in=varargin{2};
   X=varargin{1};
   Y=[1:size(in,2)];
end

opts.list=varargin(charmin:end);


if size(X,1).*size(X,2)==length(X)
   X=X(:);
end

if size(in,1).*size(in,2)==length(in)
   X=X(:);
end

if length(Y)~=size(in,2)
   error([' atldk: dimension mismatch'])
end
Y=Y(:);

if size(X,1)~=size(in,1)
   error([' atldk: dimension mismatch'])
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%interpretazione delle opzioni-------------------------------------------------
%------------------------------------------------------------------------------

%defaults
opts.pldir='vertical';
opts.color=[0,0,0]';
opts.marker=['none'];
opts.linestyle=['-'];
opts.amp='entire';
opts.gain=1;
opts.wiggle='none';
opts.clip='none';

for I=1:2:length(opts.list)

     switch opts.list{I}

     %orientamento plot
     case{'dir','direction'}

        if any(strcmp(opts.list{I+1},{'h','horiz','horizontal'}))
           opts.pldir='horizontal';
        elseif any(strcmp(opts.list{I+1},{'v','vert','vertical'}))
           opts.pldir='vertical';
        else
           error(['atldk: dir ???']);
        end

     %colori
     case{'color','col'}

        if isa(opts.list{I+1},'char')
           opts.color_str=opts.list{I+1};
           for J=1:length(opts.color_str)
              opts.color(:,J)=str2col(opts.color_str(J))';
           end
        elseif isa(opts.list{I+1},'double')
           opts.color=opts.list{I+1};
        else
           error([' atldk: color ???'])
        end

     %marker
     case{'marker'}

        if any(strcmp(opts.list{I+1},{'.','o','x','+','*','s','d','V',...
                                      '^','<','>','p','h','none'}));
           opts.marker=opts.list{I+1};
        else
           error([' atldk: marker ???']);
        end

     %linestyle
     case{'linestyle','style'}

        if any(strcmp(opts.list{I+1},{'-','--',':','-.','none','.-'}));
           opts.linestyle=opts.list{I+1};
        else
           error([' atldk: linestyle ???']);
        end

     %normalizzazione ampiezza
     case{'amp','amplitude'}

        if any(strcmp(opts.list{I+1},{'entire','e'}))
           opts.amp='entire';
        elseif any(strcmp(opts.list{I+1},{'single','s'}))
           opts.amp='single'; 
        elseif any(strcmp(opts.list{I+1},{'raw'}))
           opts.amp='raw'; 
        elseif any(strcmp(opts.list{I+1},{'agc','iagc'}))
           opts.amp='agc'; 
        else
           error(['atldk: amp ???']);
        end

      %gain
      case{'gain'}

         if isa(opts.list{I+1},'double')
           opts.gain=opts.list{I+1};
         else
           error([' atldk: gain ???'])
         end

      %wiggle
      case{'wiggle','wig'}

         if any(strcmp(opts.list{I+1},{'accurate'}))
           opts.wiggle='accurate';
         elseif any(strcmp(opts.list{I+1},{'fast'}))
           opts.wiggle='fast';
         elseif any(strcmp(opts.list{I+1},{'none'}))
           opts.wiggle='none';
         else
           error([' atldk: wiggle ???'])
         end
      
      %clip
      case{'clip'}

         if isa(opts.list{I+1},'double')
           opts.clip=opts.list{I+1};
         else
           error([' atldk: clip ???'])
         end

      otherwise

         warning(sprintf('atldk: unknown keyword %s',opts.list{I}));

      end
end

%------------------------------------------------------------------------------

   %wrapping della matrice dei colori
   indx1=size(in,2);
   indx2=size(opts.color,2);
   tmp=zeros(3,indx1);
   count=0;
   while size(tmp,2)<=indx1;
     tmp(:,count.*indx2+[1:indx2])=opts.color;
           count=count+1;
   end
   opts.color=tmp(:,1:indx1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%------------------------------------------------------------------------------
%esegue la normalizzazione e il clipping---------------------------------------

if size(in,2)>1
    val2=mean(abs(diff(Y)));
else
    val2=1; %solo una traccia
end

switch opts.amp

case 'entire'

   val1=mean(max(abs(in)),2).*ones(1,size(in,2));
   val1=val1./val2./opts.gain;
   rnorm=meshgrid(val1(:),1:size(in,1));
   
case 'single'

   val1=max(abs(in));
   val1=val1./val2./opts.gain;
   rnorm=meshgrid(val1(:),1:size(in,1));

case 'raw'

   val1=(opts.gain.^-1).*ones(1,size(in,2));
   rnorm=meshgrid(val1(:),1:size(in,1));

case 'agc'

   in=agc(in,200,'mode','center','req',1,'list',1:10:size(in,1));
   val1=mean(max(abs(in)),2).*ones(1,size(in,2));
   val1=val1./val2./opts.gain;
   rnorm=meshgrid(val1(:),1:size(in,1));

end

addval=meshgrid(Y,1:size(in,1));
in=in./rnorm;

%esegue il clipping

if eval(['strcmp(opts.clip,''none'')'],'1')
else

   indx1=find(in>(val2.*opts.clip));
   in(indx1)=val2.*opts.clip;
   indx1=find(in<(-val2.*opts.clip));
   in(indx1)=-val2.*opts.clip;

end

in=in+addval;

%------------------------------------------------------------------------------

holdstate=ishold;

if holdstate

else
   newplot;
end

hold on;

%------------------------------------------------------------------------------
%plotta i riempimenti wiggle---------------------------------------------------

if strcmp(opts.wiggle,'accurate')

      for I=1:size(in,2)
         %fprintf(1,'\r atldk wiggle: trace %u/%u running interp0',I,size(in,2))
         [in0,ax0]=interp0(in(:,I),eval('X(:,I)','X'),Y(I));
         indx1=find(in0<Y(I));
         in0(indx1)=Y(I);
         ax0=[ax0(1);ax0;ax0(end)];
         in0=[Y(I);in0;Y(I)];
         switch opts.pldir
         case 'horizontal'
            handle(size(in,2)+I)=fill(ax0,in0,'g','facecolor',opts.color(:,I),'edgecolor','none');
            %hold on
         case 'vertical'
            handle(size(in,2)+I)=fill(in0,ax0,'g','facecolor',opts.color(:,I),'edgecolor','none');
            %hold on
            axis ij
         end
      end

elseif strcmp(opts.wiggle,'fast')

      for I=1:size(in,2)
         fprintf(1,'\r atldk wiggle: trace %u/%u',I,size(in,2))
         in0=in(:,I);
         ax0=eval('X(:,I)','X');
         indx1=find(in0<Y(I));
         in0(indx1)=Y(I);
         ax0=[ax0(1);ax0;ax0(end)];
         in0=[Y(I);in0;Y(I)];
         switch opts.pldir
         case 'horizontal'
            handle(size(in,2)+I)=fill(ax0,in0,'g','facecolor',opts.color(:,I),'edgecolor','none');
            %hold on
         case 'vertical'
            handle(size(in,2)+I)=fill(in0,ax0,'g','facecolor',opts.color(:,I),'edgecolor','none');
            %hold on
            axis ij
         end
       end
   fprintf(1,'\r                                                           \r')
end

%esegue i plot-----------------------------------------------------------------

%switch opts.pldir
%case 'horizontal'
%   handle(1:size(in,2))=plot(X,in,opts.style_val);
%case 'vertical'
%   handle(1:size(in,2))=plot(in,X,opts.style_val);
%   axis ij
%end

switch opts.pldir
case 'horizontal'
   handle(1:size(in,2))=line(X,in,'linestyle',opts.linestyle,...
                                  'marker',opts.marker);
case 'vertical'
   handle(1:size(in,2))=line(in,X,'linestyle',opts.linestyle,...
                                  'marker',opts.marker,'linewidth',2.5);
   axis ij
end

%------------------------------------------------------------------------------
%modifica attributi delle curve------------------------------------------------

for I=1:size(in,2)
   set(handle(I),'color',opts.color(:,I));
end

%------------------------------------------------------------------------------

if holdstate
   hold on;
else
   hold off;
end

%------------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%subfunctions%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function col=str2col(str)

switch str

case 'r'
  col=[1,0,0];

case 'g'
  col=[0,1,0];

case 'b'
  col= [0,0,1];

case 'w'
  col=[1,1,1];

case 'm'
  col=[1,0,1];

case 'c'
  col=[0,1,1];

case 'k'
  col=[0,0,0];

case 'y'
  col=[1,1,0];

otherwise
  error([str ' is not a color string'])

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [in0,ax0]=interp0(in,ax,ref)
%[in0,ax0]=interp0(in,ax)
%interpola "in" ed il suo asse "ax" aggiungendo i campioni in cui "in" vale ref.
%INPUT
%in input. questa versione lavora sun una sola traccia di double
%ax asse dell'input
%ref 
%OUTPUT
%in0 output
%ax0 asse dell'output

ax=ax(:);
in=in(:);

ax0=ax;
indx1=find(([in(2:end)-ref].*[in(1:end-1)-ref])<0);
ax1=zeros(length(indx1),1);
for I=length(indx1):-1:1
   ax1(I)=interp1(in(indx1(I):indx1(I)+1),ax0(indx1(I):indx1(I)+1),ref,'*linear');
end
%ax0=merge(ax,ax1);
%ax0=ax0(:);
ax0=[ax;ax1(:)];
ax0=sort(ax0);
indx2=find(~diff(ax0));
ax0(indx2)=[];
in0=interp1(ax,in,ax0);
in0=in0(:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function v=merge(v1,v2)

v1=v1(:);
v2=v2(:);

if length(v1)<length(v2)
 tmp=v1;
 v1=v2;
 v2=tmp;
 clear tmp
end

v=v1;
for I=1:length(v2)
 indx1=find(v==v2(I));
 v(indx1)=[];
end
v=sort([v(:);v2(:)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
