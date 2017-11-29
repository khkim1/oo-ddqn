cd('/Users/KunhoKim/Desktop/StanfordResearch/relDQN/PG/')

fid = fopen('rewards.txt');
s1 = textscan(fid, '%f');
fclose(fid)

fid = fopen('rewards_vanilla.txt');
s2 = textscan(fid, '%f');
fclose(fid)

running_rewards = s1{1};
running_rewards_vanilla = s2{1};

figure(1)
hold on
plot(1:10:length(running_rewards)*10, running_rewards)
plot(1:10:length(running_rewards_vanilla)*10, running_rewards_vanilla)
xlabel('Number of Episodes')
ylabel('100 Episode Average Reward')
h = legend({'OOPG', 'Vanilla PG'}); 
rect = [0.725, .15, .1, .05];
set(h, 'Position', rect)

x=200;
y=200;
width=600;
height=500;
set(figure(1), 'Position', [x y width height])
a=findobj(gcf);
allaxes=findall(a,'Type','axes');
alllines=findall(a,'Type','line');
set(alllines,'Linewidth',1.5);
set(allaxes,'FontName','Helvetica','FontWeight','Bold','LineWidth',2,...
'FontSize',18,'box','on');