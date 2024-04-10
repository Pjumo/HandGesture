clear;
numADCSamples = 256; % number of ADC samples per chirp
numADCBits = 16; % number of ADC bits per sample
numTX = 3;
numRX = 4; % number of receivers
numLanes = 2; % do not change. number of lanes is always 2
isReal = 0; % set to 1 if real only data, 0 if complex data0
%% read file
% read .bin file
movement = "click";
filepath = "D:\PJM\university\Capstone\code\bin_radar\";
filename = filepath + movement + ".bin";
pngname = filepath + "image\" + movement + ".png";
fid = fopen(filename,'r');
adcData = fread(fid, 'int16');

% if 12 or 14 bits ADC per sample compensate for sign extension
if numADCBits ~= 16
    l_max = 2^(numADCBits-1)-1;
    adcData(adcData > l_max) = adcData(adcData > l_max) - 2^numADCBits;
end
fclose(fid);
fileSize = size(adcData, 1);
% real data reshape, filesize = numADCSamples*numChirps
if isReal
    numChirps = fileSize/numADCSamples/numRX;
    LVDS = zeros(1, fileSize);
    %create column for each chirp
    LVDS = reshape(adcData, numADCSamples*numRX, numChirps);
    %each row is data from one chirp
    LVDS = LVDS.';
else
    % for complex data
    % filesize = 2 * numADCSamples*numChirps
    numChirps = fileSize/2/numADCSamples/numRX;
    LVDS = zeros(1, fileSize/2);
    %combine real and imaginary part into complex data
    %read in file: 2I is followed by 2Q
    counter = 1;
    for i=1:4:fileSize-1
        LVDS(1,counter) = adcData(i) + sqrt(-1)*adcData(i+2); LVDS(1,counter+1) = adcData(i+1)+sqrt(-1)*adcData(i+3); counter = counter + 2;
    end
    % create column for each chirp
    LVDS = reshape(LVDS, numADCSamples*numRX, numChirps);
    % each row is data from one chirp
    LVDS = LVDS.';
end
%organize data per RX
adcData = zeros(numRX,numChirps*numADCSamples);
for row = 1:numRX
    for i = 1: numChirps
        adcData(row, (i-1)*numADCSamples+1:i*numADCSamples) = LVDS(i, (row-1)*numADCSamples+1:row*numADCSamples);
    end
end
% return receiver data
retVal = adcData;

Receiver_1= retVal(1,:);
Receiver_2= retVal(2,:);
Receiver_3= retVal(3,:);
Receiver_4= retVal(4,:);

frames=1000;
s=size(Receiver_2,2)/frames;
m_chirps=s/numADCSamples;
Receiver_1=reshape(Receiver_1,s,frames);
Receiver_2=reshape(Receiver_2,s,frames);
Receiver_3=reshape(Receiver_3,s,frames);
Receiver_4=reshape(Receiver_4,s,frames);
B = zeros(numADCSamples, m_chirps);
velocity_time = zeros(m_chirps, frames);
for j=1:frames
    A=Receiver_4( :, j);
    B=reshape(A, numADCSamples, m_chirps);

    doppler_main=fft2(B);
    doppler_main = fftshift(doppler_main, 2);
    doppler_main(abs(doppler_main)<2e+4) = 0;
    velocity = zeros(1, m_chirps);
    for k=1:10
        velocity = velocity + doppler_main(k, :);
    end
    velocity = reshape(velocity, m_chirps, 1);
    velocity_time(:, j) = velocity;
    

    % set(gca,'Position',[0 0 1 1]);
    % set(gca,'xtick',[],'ytick',[])
    % pngname = join(['example4_',num2str(j),'.jpg']);
    % saveas(gcf,fullfile('D:\PJM\대학\연구실\ICSL angle extraction\signal_ex\', pngname));

    % frm = getframe(1);
    % img = frame2im(frm);
    % [imind, cm] = rgb2ind(img, 256);
    % 
    % if j == 1
    %     % 첫 프레임의 경우, GIF 파일 생성
    %     imwrite(imind, cm, gifname, 'gif', 'Loopcount', 1, 'DelayTime', 0.1);
    % else
    %     % 이후 프레임의 경우, 기존 GIF 파일에 추가
    %     imwrite(imind, cm, gifname, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    % end
end

time_axis_img = (1:1000)* 0.04;
vel_axis_img=(-64:63)* 0.1201;
figure(1)
imagesc(time_axis_img, vel_axis_img, 20*log10(abs(velocity_time)))
xlabel('time in s');
ylabel('velocity in m/s');
colorbar
clim([80, 150])

saveas(gcf,fullfile(pngname));



















