clear;
numADCSamples = 256; % number of ADC samples per chirp
numADCBits = 16; % number of ADC bits per sample
numTX = 3;
numRX = 4; % number of receivers
numLanes = 2; % do not change. number of lanes is always 2
isReal = 0; % set to 1 if real only data, 0 if complex data0
%% read file
% read .bin file
fid = fopen("D:\PJM\university\Lab\dataset_0108\base.bin",'r');
adcData = fread(fid, 'int16');

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

frames=100;
s=size(Receiver_2,2)/frames;
m_chirps=s/numADCSamples/3;
Receiver_1=reshape(Receiver_1,s,frames);
Receiver_2=reshape(Receiver_2,s,frames);
Receiver_3=reshape(Receiver_3,s,frames);
Receiver_4=reshape(Receiver_4,s,frames);
doppler_before = zeros(numADCSamples, m_chirps);


for frame=1:frames
    Rx1 = Receiver_1(:,frame);
    Rx2 = Receiver_2(:,frame);
    Rx1 = reshape(Rx1, numADCSamples, 3, m_chirps);
    Rx2 = reshape(Rx2, numADCSamples, 3, m_chirps);
    
    Rx_vt = Rx1(:, 2, :);
    Rx_vt = reshape(Rx_vt, numADCSamples, m_chirps);
    doppler = fft2(Rx_vt);
    doppler = fftshift(doppler, 2);
    doppler(abs(doppler)<2e+4) = 0;
    
    doppler_res = zeros(numADCSamples, m_chirps);
    for i=1:m_chirps
        if i==33
            doppler_res(:, i) = doppler(:, i) - doppler_before(:, i);
        else
            doppler_res(:, i) = doppler(:, i);
        end
    end
    doppler_before = doppler;
    
    if frame == 1
        continue;
    end
    doppler_res_abs = abs(doppler_res);
    x_axis_img=(-32:31)* 0.1201;
    y_axis_img=(0:256)* 0.1499;
    figure(1)
    imagesc(x_axis_img, y_axis_img, doppler_res_abs)
    xlabel('velocity in m/s');
    ylabel ('range in m');
    title(['FrameID:' num2str(frame)])
    colorbar
    clim([0 3*10^5])

    % result_path = "D:\PJM\university\Lab\result_csv\two_man_walking_cross" + num2str(frame) + ".csv";
    % writematrix(doppler_res_abs, result_path)

    frm = getframe(1);
    img = frame2im(frm);
    [imind, cm] = rgb2ind(img, 256);

    if frame == 2
        % 첫 프레임의 경우, GIF 파일 생성
        imwrite(imind, cm, "D:\PJM\university\Lab\ICSL angle extraction\remove_clutter_gif\base.gif", 'gif', 'Loopcount', 1, 'DelayTime', 0.1);
    else
        % 이후 프레임의 경우, 기존 GIF 파일에 추가
        imwrite(imind, cm, "D:\PJM\university\Lab\ICSL angle extraction\remove_clutter_gif\base.gif", 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    end
end












