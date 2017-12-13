cd('/home/vision/Desktop/relDQN/DQN-object/learning_curves/trial13')

name = 'rewards5.txt';
%name = 'rewards_gpu.txt'; 

fid = fopen(name);
s1 = textscan(fid, '%d, %d, %f, %f');
fclose(fid)


fid = fopen('reward_vanilla.txt');
s2 = textscan(fid, '%d, %d, %f, %f');
fclose(fid)

window_size = 5;
window_size_v = 10; 

frames_object = s1{1}(floor(window_size/2):end-(window_size - floor(window_size/2)));
frames_vanilla = s2{1}(floor(window_size_v/2):end-floor(window_size_v/2)); 
episodes_object = s1{2}(floor(window_size/2):end-(window_size - floor(window_size/2)));
episodes_vanilla = s2{2}(floor(window_size_v/2):end-floor(window_size_v/2)); 
running_trng_rewards_object = s1{3}; 
running_trng_rewards_vanilla = s2{3}; 
running_rewards_object_raw = s1{4};
running_rewards_vanilla_raw = s2{4};

running_rewards_object = zeros(length(running_rewards_object_raw) - window_size, 1);
running_rewards_vanilla = zeros(length(running_rewards_vanilla_raw) - window_size_v, 1);
%running_rewards_vanilla = running_rewards_vanilla_raw; 

for i = window_size+1:length(running_rewards_object_raw)+1
    if i == window_size+1
        running_rewards_object(i - window_size) = mean(running_rewards_object_raw(1:window_size));
    else
        running_rewards_object(i - window_size) = (1 - 1/window_size) * running_rewards_object(i - window_size - 1) ...
                                       + (1/window_size) * running_rewards_object_raw(i - 1); 
    end
end

for i = window_size_v+1:length(running_rewards_vanilla_raw)+1
    if i == window_size_v+1
        running_rewards_vanilla(i - window_size_v) = mean(running_rewards_vanilla_raw(1:window_size_v));
    else
        running_rewards_vanilla(i - window_size_v) = (1 - 1/window_size_v) * running_rewards_vanilla(i - window_size_v - 1) ...
                                        + (1/window_size_v) * running_rewards_vanilla_raw(i - 1); 
    end
end


C = linspecer(3); 

scale = 1;

figure(1)
subplot(1, 2, 1)
hold on
grid on
plot(frames_object, scale*running_rewards_object, 'Color', C(1, :))
plot(frames_vanilla, scale*running_rewards_vanilla, 'Color', C(2, :))
xlabel('Number of Frames')
ylabel('Average Validation Reward')
h1 = legend({'OO-DDQN', 'DDQN'}); 
%rect = [0.725, .15, .1, .05];
%set(h1, 'Position', rect)
%ylim([0 ])

subplot(1, 2, 2) 
hold on
grid on
plot(episodes_object, scale*running_rewards_object, 'Color', C(1, :))
plot(episodes_vanilla, scale*running_rewards_vanilla, 'Color', C(2, :))
xlabel('Number of Episodes')
ylabel('Average Validation Reward')
h2 = legend({'OO-DDQN', 'DDQN'}); 
%rect = [0.725, .15, .1, .05];
%set(h2, 'Position', rect)
%ylim([-22, 20])



x=200;
y=200;
width=2000;
height=800;
set(figure(1), 'Position', [x y width height])
a=findobj(gcf);
allaxes=findall(a,'Type','axes');
alllines=findall(a,'Type','line');
set(alllines,'Linewidth',1.5);
set(allaxes,'FontName','Helvetica','FontWeight','Bold','LineWidth',2,...
'FontSize',18,'box','on');