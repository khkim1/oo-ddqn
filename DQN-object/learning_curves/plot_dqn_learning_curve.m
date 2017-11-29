cd('/home/vision/Desktop/relDQN/DQN-yolo/learning_curves/trial1')

name = 'rewards3.txt';
%name = 'rewards_gpu.txt'; 

fid = fopen(name);
s1 = textscan(fid, '%d, %d, %f, %f');
fclose(fid)


fid = fopen('reward_vanilla.txt');
s2 = textscan(fid, '%d, %d, %f, %f');
fclose(fid)

frames_object = s1{1}(10:end);
frames_vanilla = s2{1}; 
episodes_object = s1{2}(10:end);
episodes_vanilla = s2{2}; 
running_trng_rewards_object = s1{3}; 
running_trng_rewards_vanilla = s2{3}; 
running_rewards_object_raw = s1{4};
running_rewards_vanilla_raw = s2{4};

running_rewards_object = zeros(length(running_rewards_object_raw) - 10, 1);
%running_rewards_vanilla = zeros(length(running_rewards_vanilla_raw) - 10, 1);
running_rewards_vanilla = running_rewards_vanilla_raw; 

for i = 11:length(running_rewards_object_raw)+1
    if i == 11
        running_rewards_object(i - 10) = mean(running_rewards_object_raw(1:10));
    else
        running_rewards_object(i - 10) = 0.9 * running_rewards_object(i - 11) ...
                                       + 0.1 * running_rewards_object_raw(i - 1); 
    end
end

%{
for i = 11:length(running_rewards_vanilla_raw)+1
    if i == 11
        running_rewards_vanilla(i - 10) = mean(running_rewards_vanilla_raw(1:10));
    else
        running_rewards_vanilla(i - 10) = 0.9 * running_rewards_vanilla(i - 11) ...
                                        + 0.1 * running_rewards_vanilla_raw(i - 1); 
    end
end
%}

C = linspecer(3); 

scale = 20;

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