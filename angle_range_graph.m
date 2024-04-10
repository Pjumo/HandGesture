clear;
numADCSamples = 256; % number of ADC samples per chirp
numADCBits = 16; % number of ADC bits per sample
numTX = 3;
numRX = 4; % number of receivers
numLanes = 2; % do not change. number of lanes is always 2
isReal = 0; % set to 1 if real only data, 0 if complex data0
%% read file
% read .bin file
fid = fopen("D:\PJM\university\Lab\dataset_0108\one_man_walking_forward.bin",'r');
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
for frame=1:frames
    Rx1 = Receiver_1(:,frame);
    Rx2 = Receiver_2(:,frame);
    Rx1 = reshape(Rx1, numADCSamples, 3, m_chirps);
    Rx2 = reshape(Rx2, numADCSamples, 3, m_chirps);
    Rx1_Tx2 = Rx1(:, 2, :);
    Rx1_Tx3 = Rx1(:, 3, :);
    Rx2_Tx2 = Rx2(:, 2, :);
    Rx2_Tx3 = Rx2(:, 3, :);
    Rx_virtual = [Rx1_Tx2, Rx2_Tx2, Rx1_Tx3, Rx2_Tx3];

    Rx_virtual_doppler = zeros(4, numADCSamples, m_chirps);
    for cnt=1:4
        Rx_vt = Rx_virtual(:, cnt, :);
        Rx_vt = reshape(Rx_vt, numADCSamples, m_chirps);
        doppler = fft2(Rx_vt);
        doppler = fftshift(doppler, 2);
        doppler(abs(doppler)<2e+4) = 0;
        Rx_virtual_doppler(cnt, :, :) = doppler;
        if cnt==1
            x_axis_img=(-64:63)* 0.1201;
            y_axis_img=(0:256)* 0.1499;
            figure(1)
            imagesc(x_axis_img, y_axis_img,20*log10(abs(doppler)))
            xlabel('velocity in m/s');
            ylabel ('range in m');
            title(['FrameID:' num2str(frame)])
            colorbar
        end
    end
    
    range_angle = zeros(numADCSamples, 4);
    for x=1:numADCSamples
        peak_pos = 0;
        peak_Amp = 0;
        for y=1:m_chirps
            amp = abs(Rx_virtual_doppler(1, x, y));
            if amp ~= 0  && y ~= 33
                angle_ex = [Rx_virtual_doppler(1, x, y), Rx_virtual_doppler(2, x, y), Rx_virtual_doppler(3, x, y), Rx_virtual_doppler(4, x, y)];
                angle_ex_fft = fft(angle_ex);
                range_angle(x, :) = range_angle(x, :) + angle_ex_fft;
                % if peak_Amp<amp
                %     peak_Amp = amp;
                %     peak_pos = y;
                % end
            end
        end
        % if peak_Amp~=0
        %     angle_ex = [Rx_virtual_doppler(1, x, peak_pos), Rx_virtual_doppler(2, x, peak_pos), Rx_virtual_doppler(3, x, peak_pos), Rx_virtual_doppler(4, x, peak_pos)];
        %     angle_ex_fft = fft(angle_ex);
        %     range_angle(x, :) = angle_ex_fft;
        % end
    end
    figure(2)
    x_axis_img=(-2:2)*30;
    y_axis_img=(0:256)*0.1499;
    imagesc(x_axis_img, y_axis_img, abs(range_angle))
    xlabel('angle in degree');
    ylabel ('range in m');
    title(['FrameID:' num2str(frame)])
    colorbar
    clim([0 5*10^5])
    pause(0.05)
end


















